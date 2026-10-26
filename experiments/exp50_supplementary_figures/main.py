"""Verify saved aggregate values and redraw Supplementary Figures S6 and S7."""

from pathlib import Path

from .analysis import SOURCE_FILES, verify_and_redraw


def run(config, context):
    value = config["inputs"]["source_root"]
    if not value:
        raise ValueError("inputs.source_root is required")
    source = Path(value).expanduser()
    source = source if source.is_absolute() else context.project_root / source
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"missing source directory: {source}")
    for name in SOURCE_FILES.values():
        file = source / name
        if not file.is_file() or file.is_symlink():
            raise FileNotFoundError(f"missing input: {file}")
    output = context.result_dir(config["run"]["id"])
    result = verify_and_redraw(source, output)
    return {"status": "success", "output": str(output), "verification": result}
