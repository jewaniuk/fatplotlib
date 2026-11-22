"""the `fpl` command.

`fpl preview <script>` opens the hot-reload preview. `fpl build <script> [--target name]
[--format svg|pdf|png] [--dpi n] [--background paint]` writes `<script>.light.<format>` and
`<script>.dark.<format>` next to the script, tokens resolved to literals; a target other than the
default adds its name, as in `<script>.talk.light.svg`.
"""

import argparse
from pathlib import Path

from fatplotlib import preview, script
from fatplotlib.render.export import FORMATS, save
from fatplotlib.style import Token


def main(argv: list[str] | None = None) -> None:
    """Parses the command line and runs the subcommand.

    Args:
        argv (list[str] | None): the arguments, sys.argv if None.
    """
    parser = argparse.ArgumentParser(prog="fpl", description="academic figures with typst text")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("preview", help="preview a figure script, rebuilding on save")
    show.add_argument("script", type=Path)
    show.add_argument("--port", type=int, default=8765)
    show.add_argument("--no-browser", action="store_true", help="do not open a browser")
    build = commands.add_parser("build", help="write the light and dark files")
    build.add_argument("script", type=Path)
    build.add_argument("--target", help="a target from targets.py beside the script, paper by default")
    build.add_argument("--format", choices=FORMATS, default="svg")
    build.add_argument("--dpi", type=float, default=300, help="png resolution, 300 by default")
    build.add_argument("--background", help="a token such as bg or a #rrggbb colour, transparent by default")
    args = parser.parse_args(argv)

    if args.command == "preview":
        try:
            preview.serve(args.script, args.port, open_browser=not args.no_browser)
        except KeyboardInterrupt:
            pass
    else:
        fig, targets = script.load(args.script)
        name = script.pick(targets, args.target)
        suffix = "" if name == script.pick(targets) else f".{name}"
        background = args.background
        if background is not None and not background.startswith("#"):
            background = Token(background)
        out = args.script.with_suffix(f"{suffix}.{args.format}")
        for path in save(fig, out, targets[name], dpi=args.dpi, background=background):
            print(f"wrote {path}")
