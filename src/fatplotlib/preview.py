"""hot-reload preview: rebuild on save, swap the svg in the browser without a page reload.

one long-lived process watches the script's directory, re-runs the script on every saved `.py`
and pushes the new svg to the page over server-sent events (one open http request the server
writes to). the svg keeps tokens as css variables, and each pane's theme class defines them, so
light and dark come from one render. on an error the page shows the traceback and keeps the last
good svg.

the preview always draws panel labels (class `fpl-label`) and the page hides them when the target
has none, so `l` toggles them without a rebuild. targets re-solve the layout, so `p` asks the
server to rebuild the last loaded figure for the next target in `targets.py`.

keys: `t` cycles both / light / dark, `l` toggles panel labels, `p` cycles targets.
"""

import json
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import watchfiles

from fatplotlib import script
from fatplotlib.layout import Figure, Target
from fatplotlib.render.svg import render
from fatplotlib.theme import DARK, LIGHT

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>fpl %(name)s</title><style>
%(themes)s
html,body{margin:0;height:100%%;font:13px sans-serif}
body{display:flex;flex-direction:column;background:#808080}
.pane{flex:1;min-height:0;display:flex;padding:8px;background:var(--bg)}
.pane svg{width:100%%;height:100%%}
body[data-mode=light] .dark,body[data-mode=dark] .light{display:none}
body[data-labels=off] .fpl-label{display:none}
#status{position:fixed;right:6px;bottom:4px;color:#808080}
#error{display:none;position:fixed;inset:auto 0 0 0;max-height:50%%;overflow:auto;margin:0;
  padding:8px;background:#3a0d0d;color:#ffd6d6;white-space:pre-wrap}
</style></head>
<body data-mode="both" data-labels="on">
<div class="pane light"></div><div class="pane dark"></div>
<span id="status">connecting</span><pre id="error"></pre>
<script>
const modes = ["both", "light", "dark"];
document.addEventListener("keydown", (e) => {
  if (e.key === "t") {
    const mode = modes[(modes.indexOf(document.body.dataset.mode) + 1) %% modes.length];
    document.body.dataset.mode = mode;
  } else if (e.key === "l") {
    document.body.dataset.labels = document.body.dataset.labels === "on" ? "off" : "on";
  } else if (e.key === "p") {
    fetch("/target", { method: "POST" });
  }
});
let target = null;
const events = new EventSource("/events");
events.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  const error = document.getElementById("error");
  if (msg.svg !== null) {
    // each pane gets its own ids: gradients resolve css variables where they are defined
    document.querySelectorAll(".pane").forEach((pane) => {
      const prefix = pane.classList[1] + "-";
      pane.innerHTML = msg.svg
        .replaceAll(' id="', ' id="' + prefix)
        .replaceAll("url(#", "url(#" + prefix)
        .replaceAll('href="#', 'href="#' + prefix);
    });
  }
  if (msg.target !== null && msg.target !== target) {
    // a new target sets labels to its own default; l overrides until the next switch
    target = msg.target;
    document.body.dataset.labels = msg.labels ? "on" : "off";
  }
  error.textContent = msg.error ?? "";
  error.style.display = msg.error ? "block" : "none";
  document.getElementById("status").textContent = msg.status;
};
events.onerror = () => (document.getElementById("status").textContent = "disconnected");
</script></body></html>
"""


class _State:
    """The latest build, shared between the watcher and the http threads.

    Attributes:
        changed (threading.Condition): notified after every build.
        version (int): counts builds, so a waiting client knows when there is news.
        message (str): the latest build as json: svg (last good, or null), error, status, target
            name (or null) and whether that target draws labels.
        building (threading.Lock): held while a build runs, so saves and `p` do not overlap.
        figure (Figure | None): the last loaded figure, rebuilt by `p` without re-running the script.
        targets (dict[str, Target]): the targets loaded with it.
        target (str | None): the name of the target being previewed.
    """

    def __init__(self) -> None:
        self.changed = threading.Condition()
        self.version = 0
        self.message = json.dumps({"svg": None, "error": None, "status": "building", "target": None, "labels": True})
        self.building = threading.Lock()
        self.figure: Figure | None = None
        self.targets: dict[str, Target] = {}
        self.target: str | None = None
        self._svg: str | None = None

    def publish(self, svg: str | None, error: str | None, status: str) -> None:
        """Stores a build and wakes every waiting client.

        Args:
            svg (str | None): the new svg, None to keep the last good one.
            error (str | None): the traceback, None if the build worked.
            status (str): a short line for the corner of the page.
        """
        labels = self.targets[self.target].labels if self.target in self.targets else True
        with self.changed:
            self._svg = svg or self._svg
            self.message = json.dumps(
                {"svg": self._svg, "error": error, "status": status, "target": self.target, "labels": labels}
            )
            self.version += 1
            self.changed.notify_all()


def serve(path: Path, port: int = 8765, open_browser: bool = True) -> None:
    """Previews a figure script until interrupted.

    Args:
        path (Path): the script.
        port (int): the local http port.
        open_browser (bool): open the page in the default browser.
    """
    state = _State()
    page = (_PAGE % {"name": path.name, "themes": LIGHT.css(".light") + "\n" + DARK.css(".dark")}).encode()
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(page, state, path))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    print(f"previewing {path.name} at {url}")
    if open_browser:
        webbrowser.open(url)
    _build(path, state)
    for _ in watchfiles.watch(path.resolve().parent, watch_filter=watchfiles.PythonFilter()):
        _build(path, state)


def _build(path: Path, state: _State, next_target: bool = False) -> None:
    """Runs the script (or, for `p`, moves to the next target) and publishes the svg or the error."""
    with state.building:
        start = time.perf_counter()
        try:
            if next_target and state.figure is not None:
                names = list(state.targets)
                state.target = names[(names.index(state.target) + 1) % len(names)] if state.target in names else names[0]
            else:
                state.figure, state.targets = script.load(path)
                kept = state.target if state.target in state.targets else None
                state.target = kept or script.pick(state.targets)
            assert state.figure is not None and state.target is not None  # loaded above or before
            svg = render(*state.figure.scene(state.targets[state.target].with_(labels=True)))
        except Exception:
            error = traceback.format_exc()
            print(error)
            state.publish(None, error, "error")
            return
        ms = (time.perf_counter() - start) * 1000
        print(f"built {state.target} in {ms:.0f} ms")
        state.publish(svg, None, f"{state.target} {time.strftime('%H:%M:%S')} {ms:.0f} ms")


def _handler(page: bytes, state: _State, path: Path) -> type[BaseHTTPRequestHandler]:
    """A request handler serving the page at /, the build stream at /events and the next target
    on a post to /target."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/target":
                self.send_error(404)
                return
            self.send_response(204)
            self.end_headers()
            _build(path, state, next_target=True)

        def do_GET(self) -> None:
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page)
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self._stream()
            else:
                self.send_error(404)

        def _stream(self) -> None:
            """Sends the current build, then each new one, until the page closes."""
            seen = -1
            try:
                while True:
                    with state.changed:
                        fresh = state.changed.wait_for(lambda: state.version != seen, timeout=15)
                        seen, message = state.version, state.message
                    # on a timeout send an sse comment, which keeps the connection alive
                    self.wfile.write(f"data: {message}\n\n".encode() if fresh else b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            """Keeps request lines out of the terminal."""

    return Handler
