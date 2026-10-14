# PaperLeet brand assets

The PaperLeet mark is a front-facing black cat head with pointed ears and gold eyes. Its cream circular badge keeps the cat visible on both light and dark backgrounds. The geometry and colors match `web/src/Brand.tsx`.

| Asset | Canvas | Use |
| --- | --- | --- |
| `paperleet-logo-light.svg` | 320 × 80 | Full wordmark on light backgrounds; transparent canvas |
| `paperleet-logo-dark.svg` | 320 × 80 | Full wordmark on dark backgrounds; transparent canvas |
| `paperleet-mark.svg` | 96 × 96 | Standalone black cat on its cream circular badge |
| `paperleet-cat.svg` | 96 × 96 | Black cat head on a transparent canvas; use on light backgrounds |
| `../favicon.svg` | 64 × 64 | Browser icon with the same cream badge |
| `paperleet-brand-preview.svg` | 1200 × 640 | Light and dark logo sheet with 96, 32, and 16 px icon samples |

| Color | Value |
| --- | --- |
| Black cat | `#161619` |
| Inner ears | `#3b3b40` |
| Gold eyes | `#e8c77b` |
| Cream badge | `#f3efe6` |
| Nose | `#b9b6af` |
| Mouth | `#77757a` |
| Light-theme wordmark | `#262626` |
| Dark-theme wordmark | `#efeff1` |

The cat stays black in both interface themes. Keep the cream badge on dark backgrounds; do not invert the head to white. SVG assets contain native paths and shapes with no external references or runtime fonts. Wordmark and preview-label glyphs are outlined from Arial Bold; the website header uses its system font stack.
