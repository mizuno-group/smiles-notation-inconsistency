"""Generic, strict dispatcher for repository experiment entrypoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

from .experiment import ExperimentContext

_EXPERIMENT_NAME_PATTERN = re.compile(r"exp[0-9]{2,3}_[a-z][a-z0-9_]*")


def load_config(path: Path) -> dict[str, Any]:
    """Load a TOML mapping from an experiment configuration file."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"configuration must be a regular file: {path}")
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"configuration root must be a table: {path}")
    return value


def _parse_override(override: str) -> tuple[tuple[str, ...], Any]:
    if "=" not in override:
        raise ValueError(f"override must be dotted.key=TOML_VALUE, got {override!r}")
    raw_key, raw_value = override.split("=", maxsplit=1)
    parts = tuple(raw_key.split("."))
    if not parts or any(not part for part in parts):
        raise ValueError(f"override contains an empty key component: {override!r}")
    try:
        value = tomllib.loads(f"value = {raw_value}")["value"]
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"override value must be a TOML literal: {override!r}: {exc}") from exc
    return parts, value


def _set_existing_key(config: dict[str, Any], parts: tuple[str, ...], value: Any) -> None:
    current: dict[str, Any] = config
    for part in parts[:-1]:
        if part not in current:
            raise KeyError(f"override path does not exist: {'.'.join(parts)}")
        nested = current[part]
        if not isinstance(nested, dict):
            raise TypeError(f"override path is not a table: {'.'.join(parts[:-1])}")
        current = nested

    leaf = parts[-1]
    if leaf not in current:
        raise KeyError(f"override key does not exist: {'.'.join(parts)}")
    existing = current[leaf]
    if type(value) is not type(existing):
        raise TypeError(
            f"override type differs for {'.'.join(parts)}: "
            f"expected {type(existing).__name__}, got {type(value).__name__}"
        )
    current[leaf] = value


def apply_overrides(config: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply strict dotted-key overrides to existing configuration keys."""

    resolved = copy.deepcopy(dict(config))
    for override in overrides:
        parts, value = _parse_override(override)
        _set_existing_key(resolved, parts, value)
    return resolved


def resolve_experiment_dir(project_root: Path, experiment: str) -> Path:
    """Resolve a stage-named experiment directory under ``experiments/``."""

    if not _EXPERIMENT_NAME_PATTERN.fullmatch(experiment):
        raise ValueError(
            "experiment must be a directory name such as exp00_prepare_data or "
            f"exp10_train, got {experiment!r}"
        )
    experiments_root = (project_root / "experiments").resolve()
    unresolved = experiments_root / experiment
    if unresolved.is_symlink():
        raise ValueError(f"experiment directory must not be a symbolic link: {unresolved}")
    candidate = unresolved.resolve()
    if candidate.parent != experiments_root:
        raise ValueError(f"experiment path escapes experiments root: {candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {candidate}")
    return candidate


def resolve_config_path(
    project_root: Path,
    experiment_dir: Path,
    config: Path | None,
) -> Path:
    """Resolve a regular configuration file contained by the experiment directory."""

    if config is None:
        unresolved = experiment_dir / "config.toml"
    else:
        if config.is_absolute() or ".." in config.parts:
            raise ValueError("configuration path must be repository-relative without '..'")
        unresolved = (
            project_root / config
            if config.parts and config.parts[0] == "experiments"
            else experiment_dir / config
        )
    if unresolved.is_symlink():
        raise ValueError(f"configuration must not be a symbolic link: {unresolved}")
    candidate = unresolved.resolve()
    experiment_root = experiment_dir.resolve()
    if experiment_root != candidate and experiment_root not in candidate.parents:
        raise ValueError(f"configuration is outside the experiment directory: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"configuration must be a regular file: {candidate}")
    return candidate


def _clear_experiment_modules(package_name: str) -> None:
    for loaded_name in tuple(sys.modules):
        if loaded_name == package_name or loaded_name.startswith(f"{package_name}."):
            del sys.modules[loaded_name]


def load_experiment_module(experiment_dir: Path) -> ModuleType:
    """Load an isolated experiment package and require ``run(config, context)``."""

    main_path = experiment_dir / "main.py"
    if not main_path.is_file() or main_path.is_symlink():
        raise FileNotFoundError(f"experiment entrypoint is missing: {main_path}")

    identity = hashlib.sha256(str(experiment_dir.resolve()).encode()).hexdigest()[:12]
    package_name = f"_research_experiment_{identity}_{experiment_dir.name}"
    module_name = f"{package_name}.main"
    _clear_experiment_modules(package_name)

    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(experiment_dir)]
    package.__spec__ = ModuleSpec(package_name, loader=None, is_package=True)
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        _clear_experiment_modules(package_name)
        raise RuntimeError(f"unable to load experiment module: {main_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        _clear_experiment_modules(package_name)
        raise
    if not callable(getattr(module, "run", None)):
        _clear_experiment_modules(package_name)
        raise AttributeError(f"{main_path} must define callable run(config, context)")
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one tracked experiment through its experiment-specific entrypoint."
    )
    parser.add_argument("experiment", help="stage directory name under experiments/")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="strict existing-key overrides in dotted.key=TOML_VALUE form",
    )
    parser.add_argument("--config", type=Path, help="config path inside the experiment")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="output root; relative paths are resolved from the project root",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print only")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path,
) -> int:
    """Resolve configuration and dispatch to one experiment entrypoint."""

    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {project_root}")
    args = build_parser().parse_intermixed_args(argv)
    experiment_dir = resolve_experiment_dir(project_root, args.experiment)
    config_path = resolve_config_path(project_root, experiment_dir, args.config)
    config = apply_overrides(load_config(config_path), args.overrides)
    results_root = (
        args.results_root.resolve()
        if args.results_root.is_absolute()
        else (project_root / args.results_root).resolve()
    )
    context = ExperimentContext(
        project_root=project_root,
        experiment_dir=experiment_dir,
        experiment_name=experiment_dir.name,
        config_path=config_path,
        results_root=results_root,
        overrides=tuple(args.overrides),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": config,
                    "config_path": str(config_path.relative_to(project_root)),
                    "experiment": experiment_dir.name,
                    "results_root": str(results_root),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    result = load_experiment_module(experiment_dir).run(config, context)
    if not isinstance(result, Mapping):
        raise TypeError(
            f"experiment {experiment_dir.name} returned {type(result).__name__}; expected a mapping"
        )
    print(json.dumps(dict(result), allow_nan=False, indent=2, sort_keys=True))
    return 0
