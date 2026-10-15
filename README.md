# PaperLeet Offline

**English** | [简体中文](README.zh-CN.md)

Read papers, implement their core methods, and run the tests on your own CPU. This standalone edition contains the paper library, Python editor, local solutions, test reports, and teaching experiments. There are no accounts, profiles, community, online presence, or cloud submissions.

The curriculum follows [PaperLeet's public backbone](https://github.com/PaperLeet/paperleet.github.io). This release starts with **49 papers and 63 exercises**; `sync/` records the upstream revision and managed files.

## Start locally

Requires Python 3.10+ and Node.js 20.19+. Dependency installation needs internet access; ordinary practice after installation uses only your computer.

```bash
git clone https://github.com/Hongyang-Du/PaperLeet.git
cd PaperLeet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
npm --prefix web ci
make dev
```

Open **http://127.0.0.1:5173**, choose a paper and module, complete the required functions, and run the tests. Python/PyTorch runs on your computer; no GPU or account is needed. Paper links access external sites only when you choose to open them.

Run `make dev` again to restart. **Ctrl-C stops both services.** The judge binds to `127.0.0.1` and executes one job at a time. Submitted Python runs with your local user's permissions; use it for your own practice code.

## Use your editor or terminal

```bash
paperleet list
paperleet init architecture/rms_norm
paperleet test architecture/rms_norm
```

`workspace/` holds local solutions and `reports/` holds results. Both are ignored by Git. Keep them when updating. The web editor also keeps browser recovery drafts; it never sends your code to the online PaperLeet service.

## Curriculum updates

The [upstream sync workflow](.github/workflows/sync-upstream.yml) checks daily and supports manual runs from **Actions**. Curriculum changes create or update one pull request after passing the offline checks. It synchronizes question packages, paper metadata, required judge helpers, and supplied experiment data. The offline UI and local service remain independently maintained here.

See [the sync guide](docs/upstream-sync.md). After an update is merged, run `git pull` and restart `make dev`.

## Contribute and check

Question fixes normally belong in the [public backbone](https://github.com/PaperLeet/paperleet.github.io), so both editions receive them. Offline UI and local-service fixes belong here. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
make check   # Framework and frontend checks
make build   # Export the catalog and build the frontend
make ci      # Also check references, mutants, and CPU budgets
```

Earlier Git history contains the former combined application. The current tree is the standalone offline edition.

## License

[MIT](LICENSE). Retain the existing source attributions and third-party notices in question packages.
