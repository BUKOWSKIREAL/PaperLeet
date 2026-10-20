# Contributing to PaperLeet Offline

The owner-assisted organization transfer and future public contribution entry
point are described in [the repository migration plan](docs/repository-migration.md).
Until public core consolidation is verified, keep the existing upstream sync
and contribution routing below.

This repository maintains the offline interface and local CPU service. Shared questions, catalog metadata, and judge helpers are managed by the upstream sync workflow.

- For a new paper, a corrected test, or a reference implementation, submit a change to [PaperLeet/paperleet.github.io](https://github.com/PaperLeet/paperleet.github.io). The offline sync PR brings reviewed changes here.
- For offline navigation, local drafts, installation, or native judge integration, submit a change here.
- Keep long-lived local question modifications on a separate branch. The sync tool stops on changes to managed files instead of overwriting them.

Question packages keep bilingual statements, starter and reference implementations, layered tests (T0 interface, T1 reference, T2 properties, T3 behavior), and reproducible CPU experiments. Preserve provenance and third-party attribution in each package. Keep module IDs stable so existing solutions remain addressable.

Use `make check` for framework/frontend checks, `make build` for the generated website, and `make ci` for reference, mutant, and CPU-budget validation. Do not commit `workspace/`, `reports/`, database files, credentials, or generated personal data.

The offline application starts without an account or external API. Keep the service loopback-only, limit concurrent judging, and preserve unsaved drafts when requests fail. Dependencies are installed during setup; ordinary practice must not download runtimes.
