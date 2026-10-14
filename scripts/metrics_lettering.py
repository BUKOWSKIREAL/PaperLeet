"""Self-contained handwritten lettering for SVG images rendered on GitHub.

The bundled outlines are derived from Patrick Hand under the SIL OFL 1.1.
Keeping glyphs as paths avoids browser font substitution and requires no font
package at generation time. See assets/metrics-lettering for source and license.
"""
from __future__ import annotations

from functools import lru_cache
from html import escape
import json
from pathlib import Path
import re


ASSETS = Path(__file__).resolve().parent / "assets" / "metrics-lettering"
PREFIX = "pl-hand-"


@lru_cache(maxsize=1)
def _font() -> dict:
    return json.loads((ASSETS / "glyphs.json").read_text(encoding="utf-8"))


def _glyphs(value: object) -> list[tuple[str, dict]]:
    glyphs = _font()["glyphs"]
    return [(character if character in glyphs else "?", glyphs.get(character, glyphs["?"])) for character in str(value)]


def text_width(value: object, size: float = 12) -> float:
    """Advance width in SVG pixels; matches text() including fallback glyphs."""
    return sum(glyph["advance"] for _, glyph in _glyphs(value)) * size / _font()["units_per_em"]


def lettering_defs() -> str:
    """Emit once per SVG; outlines and license are embedded without external URLs."""
    license_text = (ASSETS / "OFL.txt").read_text(encoding="utf-8")
    parts = [f"<metadata>{escape(license_text)}</metadata>", '<defs aria-hidden="true">']
    for character, glyph in _font()["glyphs"].items():
        if glyph["path"]:
            parts.append(f'<path id="{PREFIX}{ord(character):x}" d="{escape(glyph["path"], quote=True)}"/>')
    parts.append("</defs>")
    return "\n".join(parts)


def text(x: float, y: float, value: object, fill: str, size: float = 12, extra: str = "") -> str:
    """Draw baseline-aligned lettering, with SVG text anchors and transforms.

    ``extra`` accepts trusted SVG attributes, as in the chart renderer's former
    <text> helper. ``text-anchor`` is implemented using actual glyph advances;
    a supplied transform (for example a y-axis rotation) wraps the positioned
    lettering. The original string remains available to assistive technology.
    """
    value = str(value)
    anchor_match = re.search(r'\btext-anchor\s*=\s*([\'"])(start|middle|end)\1', extra)
    anchor = anchor_match.group(2) if anchor_match else "start"
    if anchor_match:
        extra = extra[:anchor_match.start()] + extra[anchor_match.end():]
    width = text_width(value, size)
    start = x - (width / 2 if anchor == "middle" else width if anchor == "end" else 0)
    scale = size / _font()["units_per_em"]
    parts = [
        f'<g fill="{escape(fill, quote=True)}" role="img" aria-label="{escape(value, quote=True)}" {extra.strip()}>',
        f'<title>{escape(value)}</title>',
        f'<g transform="translate({start:g} {y:g}) scale({scale:g} {-scale:g})" aria-hidden="true">',
    ]
    advance = 0
    for character, glyph in _glyphs(value):
        if glyph["path"]:
            parts.append(f'<use href="#{PREFIX}{ord(character):x}" x="{advance:g}"/>')
        advance += glyph["advance"]
    parts.append("</g></g>")
    return "".join(parts)
