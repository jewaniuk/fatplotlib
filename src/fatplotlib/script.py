"""running a figure script.

a script sets a module-level `fig` (a `Figure`) and never mentions targets. targets live in a
`targets.py` beside the script, as module-level `Target`s (`paper = Target(...)`); with no such
file there is one built-in default. the script runs in this process, so the typst compiler and
text cache stay warm between preview rebuilds.
"""

import importlib
import runpy
import sys
from pathlib import Path

from fatplotlib.layout import Figure, Target

# the target picked when none is named, if targets.py defines it
DEFAULT = "paper"


def load(path: Path) -> tuple[Figure, dict[str, Target]]:
    """Runs a figure script and reads the targets beside it.

    Modules imported from the script's directory (targets, user components) are dropped from the
    import cache first, so edits to them take effect on the next run.

    Args:
        path (Path): the script.

    Returns:
        tuple[Figure, dict[str, Target]]: the figure, and the targets by name in the order
            `targets.py` defines them ({"default": Target()} without one).

    Raises:
        TypeError: the script does not set `fig` to a `Figure`.
    """
    directory = path.resolve().parent
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file is not None and Path(file).resolve().is_relative_to(directory):
            del sys.modules[name]
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    names = runpy.run_path(str(path), run_name="__main__")
    if not isinstance(names.get("fig"), Figure):
        raise TypeError(f"{path.name} must set fig to a Figure")
    targets = {"default": Target()}
    if (directory / "targets.py").exists():
        module = importlib.import_module("targets")
        targets = {k: v for k, v in vars(module).items() if isinstance(v, Target)} or targets
    return names["fig"], targets


def pick(targets: dict[str, Target], name: str | None = None) -> str:
    """The name of the target to build.

    Args:
        targets (dict[str, Target]): the targets by name.
        name (str | None): the requested name; if None, "paper" when defined, else the first.

    Returns:
        str: the name.

    Raises:
        KeyError: the requested target is not defined.
    """
    if name is None:
        return DEFAULT if DEFAULT in targets else next(iter(targets))
    if name not in targets:
        raise KeyError(f"no target {name!r}, choose from {', '.join(targets)}")
    return name
