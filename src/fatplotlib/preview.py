"""hot-reload preview: rebuild on save, swap the svg in the browser without a page reload.

one long-lived process watches the script's directory, re-runs the script on every saved `.py`
and pushes the new svg to the page over server-sent events (one open http request the server
writes to). the svg keeps tokens as css variables, and each pane's theme class defines them, so
light and dark come from one render. on an error the page shows the traceback and keeps the last
good svg.

keys: `t` cycles both / light / dark.
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
#status{position:fixed;right:6px;bottom:4px;color:#808080}
#error{display:none;position:fixed;inset:auto 0 0 0;max-height:50%%;overflow:auto;margin:0;
  padding:8px;background:#3a0d0d;color:#ffd6d6;white-space:pre-wrap}
</style></head>
<body data-mode="both">
<div class="pane light"></div><div class="pane dark"></div>
<span id="status">connecting</span><pre id="error"></pre>
<script>
const modes = ["both", "light", "dark"];
document.addEventListener("keydown", (e) => {
  if (e.key === "t") {
    const mode = modes[(modes.indexOf(document.body.dataset.mode) + 1) %% modes.length];
    document.body.dataset.mode = mode;
  }
});
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
        message (str): the latest build as json: svg (last good, or null), error and status.
    """

    def __init__(self) -> None:
        self.changed = threading.Condition()
        self.version = 0
        self.message = json.dumps({"svg": None, "error": None, "status": "building"})
        self._svg: str | None = None

    def publish(self, svg: str | None, error: str | None, status: str) -> None:
        """Stores a build and wakes every waiting client.

        Args:
            svg (str | None): the new svg, None to keep the last good one.
            error (str | None): the traceback, None if the build worked.
            status (str): a short line for the corner of the page.
        """
        with self.changed:
            self._svg = svg or self._svg
            self.message = json.dumps({"svg": self._svg, "error": error, "status": status})
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
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(page, state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    print(f"previewing {path.name} at {url}")
    if open_browser:
        webbrowser.open(url)
    _build(path, state)
    for _ in watchfiles.watch(path.resolve().parent, watch_filter=watchfiles.PythonFilter()):
        _build(path, state)


def _build(path: Path, state: _State) -> None:
    """Runs the script once and publishes the svg or the error."""
    start = time.perf_counter()
    try:
        svg = render(*script.load(path))
    except Exception:
        error = traceback.format_exc()
        print(error)
        state.publish(None, error, "error")
        return
    ms = (time.perf_counter() - start) * 1000
    print(f"built in {ms:.0f} ms")
    state.publish(svg, None, f"{time.strftime('%H:%M:%S')} {ms:.0f} ms")


def _handler(page: bytes, state: _State) -> type[BaseHTTPRequestHandler]:
    """A request handler serving the page at / and the build stream at /events."""

    class Handler(BaseHTTPRequestHandler):
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
