"""`synthpipe <command>` dispatcher.

Each command lives in its own module exposing ``add_arguments(parser)`` and
``run(args) -> int``. Modules are imported lazily so one stage's missing
dependency never breaks the others.
"""

import argparse
import importlib
import sys

COMMANDS = {
    "corpus": ("synthpipe.corpus", "Download and filter the STL corpus"),
    "slice": ("synthpipe.slicing", "Slice corpus models into G-code for each printer profile"),
    "timeline": ("synthpipe.gcode", "Parse G-code into a Timeline .npz"),
    "realdata": ("synthpipe.realdata", "Rebuild de-duplicated, grouped real-data splits"),
    "sample": ("synthpipe.sampler", "Sample job specs into a run manifest"),
    "run": ("synthpipe.runner", "Run Blender render workers + post-processing (resumable)"),
    "status": ("synthpipe.runner_status", "Show run progress"),
    "post": ("synthpipe.postprocess", "Post-process rendered jobs (ISP, masks, labels)"),
}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="synthpipe")
    sub = parser.add_subparsers(dest="command", required=True)
    if argv and argv[0] in COMMANDS:
        module_name, help_text = COMMANDS[argv[0]]
        module = importlib.import_module(module_name)
        module.add_arguments(sub.add_parser(argv[0], help=help_text))
    else:
        for name, (_, help_text) in COMMANDS.items():
            sub.add_parser(name, help=help_text)
    args = parser.parse_args(argv)
    module = importlib.import_module(COMMANDS[args.command][0])
    return int(module.run(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
