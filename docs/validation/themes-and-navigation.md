# Themes and navigation validation

Validated on 2026-10-13.

The library now opens with categories, search, filters, paper rows, and a compact module-progress count. Promotional headings, duplicate introductions, and the library sidebar were removed. Paper details retain the technical summary and expose practice scope through a disclosure control. The community opens with its title, board navigation, search, and posting action.

Light and Dark are explicit, labeled buttons with selected state. Initial appearance follows the system unless a valid browser preference exists. Selection updates page surfaces, CodeMirror, Shiki examples, and chart colors. The document bootstrap sets the theme before the app renders.

## Checks

- `npm --prefix web run check`: all 35 problem payloads, visualization contracts, dual-theme syntax highlighting, practice isolation, and account checks passed.
- `npm --prefix web run build`: TypeScript and production bundle passed.
- `git diff --check`: passed.
- Chrome: light selection survived reload; both theme buttons updated their selected state and page colors.
- Chrome: search for Kimi returned one paper; Architecture returned four papers; Hard difficulty returned three papers; reset restored 26 papers.
- Chrome: keyboard focus is visible on filter controls; Enter expands the paper's practice scope; module links open the expected workspace.
- Chrome: an already-rendered reference solution changed its background, foreground, and token colors immediately when switching themes. The editor and reference toy charts were inspected in light and dark appearances.
- Chrome: English and Chinese layouts checked at desktop width and 390 px; navigation also checked at 901 and 1024 px. Temporary viewport overrides were reset.
- Chrome: community navigation and empty-state rendering inspected in both appearances. No posts, accounts, or judge submissions were created for this UI check.

This change affects presentation and browsing controls. It does not change problem definitions, judging, account storage, or deployment configuration.
