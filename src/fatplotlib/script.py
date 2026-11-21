"""running a figure script.

until `Figure` exists (stage 4), a script sets three module-level names: `root` (the scene),
`width` and `height` (pt). the script runs in this process, so the typst compiler and text cache
stay warm between preview rebuilds.
"""

import runpy
import sys
from pathlib import Path

from fatplotlib.scene import Node


def load(path: Path) -> tuple[Node, float, float]:
    """Runs a figure script and returns what it built.

    Modules imported from the script's directory (targets, user components) are dropped from the
    import cache first, so edits to them take effect on the next run.

    Args:
        path (Path): the script.

    Returns:
        tuple[Node, float, float]: the scene, page width and page height in pt.

    Raises:
        NameError: the script does not set `root`, `width` and `height`.
    """
    directory = path.resolve().parent
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file is not None and Path(file).resolve().is_relative_to(directory):
            del sys.modules[name]
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    names = runpy.run_path(str(path), run_name="__main__")
    missing = [name for name in ("root", "width", "height") if name not in names]
    if missing:
        raise NameError(f"{path.name} must set {', '.join(missing)}")
    return names["root"], names["width"], names["height"]
