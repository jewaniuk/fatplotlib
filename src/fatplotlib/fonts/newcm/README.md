# new computer modern sans

the default font set: sans text and sans math in the computer modern design.

- `NewCMSans10-{Regular,Bold,Oblique,BoldOblique}.otf`: text
- `NewCMSansMath-Regular.otf`: math, an opentype font with a MATH table, so it works in typst and in lualatex with `unicode-math`

copied unmodified from the `newcomputermodern` package in texlive 2026 (`texlive-fontsextra 2026.1`, `/usr/share/texmf-dist/fonts/opentype/public/newcomputermodern/`). released under the GUST Font License (`GUST-FONT-LICENSE.txt`), which is the latex project public license 1.3c with a request to rename modified fonts.

to match these fonts in a lualatex paper:

```latex
\usepackage{unicode-math}
\setsansfont{NewCMSans10-Regular.otf}[
  BoldFont=NewCMSans10-Bold.otf,
  ItalicFont=NewCMSans10-Oblique.otf,
  BoldItalicFont=NewCMSans10-BoldOblique.otf]
\renewcommand{\familydefault}{\sfdefault}
\setmathfont{NewCMSansMath-Regular.otf}
```
