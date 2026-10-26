"""Output-directory support for publication experiments."""

from __future__ import annotations


def new_output(config, context):
    output = context.result_dir(config["run"]["id"])
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; use another run.id")
    output.mkdir(parents=True)
    return output
