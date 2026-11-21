// fatplotlib typst preamble, prepended to every snippet

// font set: new computer modern sans for text and math, bundled in fonts/newcm
#set text(font: "New Computer Modern Sans")
#show math.equation: set text(font: "New Computer Modern Sans Math")

// dirac notation, mimicking the latex braket package
// delimiters grow with their content, like \left| x \right\rangle
#let ket(x) = $lr(| #x chevron.r)$
#let bra(x) = $lr(chevron.l #x |)$
#let braket(x, y) = $lr(chevron.l #x mid(|) #y chevron.r)$

// latex \ldots: three spaced periods, wider than typst's single ellipsis glyph `dots`
#let ldots = $. thin . thin .$
