# fatplotlib

git mode: main

python library (`import fatplotlib as fpl`, cli `fpl`) for academic figures that mix schematics, diagrams and data plots on one canvas, with typst for text and math, svg output, a hot-reload preview, light and dark themes, and auto-placed panel labels.

- the design plan lives in `planning/PLAN.md` (local only, gitignored). read the sections a task needs rather than the whole file
- `uv` manages the environment: `uv run`, `uv add`
- code mode is mostly agent. after each change, end with a read-in-order walkthrough: files and functions added or changed, in the order to review them in neovim, with one line each on why they sit where they do
