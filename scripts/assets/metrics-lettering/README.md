# Handwritten chart lettering

`glyphs.json` contains SVG outlines for printable ASCII and the punctuation used
by the README charts, converted from **Patrick Hand Regular** by Patrick
Wagesreiter. These outlines are distributed under the SIL Open Font License 1.1
in [OFL.txt](OFL.txt). The font's official Google Fonts metadata labels it `OFL`.

- Upstream: https://github.com/google/fonts/tree/main/ofl/patrickhand
- Font: https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf
- Original font SHA-256: `0f173b3e6cb6d1af25babf7f0057c5ac4ee11f9992b0469bb817e967ef4ad0fc`

The chart generator uses only Python's standard library. It emits reusable glyph
paths and `<use>` elements, so GitHub's image renderer needs no installed font,
external request, embedded font file, or JavaScript. Each generated SVG includes
the font copyright and complete license in readable `<metadata>`.

To regenerate outlines during development, load the upstream TTF with
`fontTools.ttLib.TTFont`, look up each character in `font.getBestCmap()`, draw the
glyph with `fontTools.pens.svgPathPen.SVGPathPen(font.getGlyphSet())`, and store its
`getCommands()` result and advance width. Preserve the font's `head.unitsPerEm`
in the JSON. `fontTools` is only a conversion tool and is not a project runtime
dependency. Unsupported characters render as `?`; accessibility labels retain
the complete original string.
