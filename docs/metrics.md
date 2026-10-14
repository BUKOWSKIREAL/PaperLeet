# README metrics

The English and Chinese READMEs display the same aggregate data with light and dark, hand-drawn SVG charts. The lettering is bundled as vector outlines so it renders consistently on GitHub; see the [font source and license](../scripts/assets/metrics-lettering/README.md). Pen-style axes are decorative; plotted values retain their exact coordinates. These files contain dates and counts, never account identities or credentials.

| Chart | Definition | Source |
| --- | --- | --- |
| GitHub stars | Cumulative star dates of the repository's **current stargazers**. GitHub does not provide the dates of removed stars, so this is not a complete record of historical star totals. | GitHub's [stargazers endpoint](https://docs.github.com/en/rest/activity/starring#list-stargazers), with timestamp media type and pagination. |
| Created accounts | Cumulative retained account rows by creation date in UTC. Includes unused administrator-created beta accounts and disabled accounts. This is **not** a visitor, activated-user, or active-user count. Hard-deleted rows no longer contribute. | The website's public `GET /api/public/metrics` endpoint. |
| Weekly commits | Non-merge commits in the checked-out default branch, grouped into weeks starting Monday in UTC. Commits that only update `docs/metrics/` are excluded. The current week is incomplete. | Local Git history, fetched in full by the workflow. |

The account endpoint queries only creation dates and aggregate counts, through a read-only database connection. It exposes no usernames, emails, access codes, IP addresses, institutions, posts, drafts, or submissions. A missing or invalid database produces an error rather than a misleading zero.

## Refresh

[README metrics](../.github/workflows/readme-metrics.yml) runs daily at 17:17 UTC, on relevant script/workflow changes, or through **Actions → README metrics → Run workflow**. GitHub schedules may be delayed. A successful run commits only generated files under `docs/metrics/`; this workflow does not deploy the website or modify its database. The workflow is enabled only for the upstream repository's `main` branch.

Each chart displays its source timestamp. If the beta server is asleep or offline, the generator keeps the last successful account snapshot and marks it as unavailable/stale. With no previous snapshot, it shows a no-data state. It never substitutes a zero or extrapolates user growth. Stars use the same retained-snapshot behavior when GitHub is unavailable.

To regenerate locally from the repository root:

```bash
python scripts/readme_metrics.py \
  --repo Hongyang-Du/PaperLeet \
  --users-url https://paperleet.com/api/public/metrics \
  --output-dir docs/metrics
```

The script uses Python's standard library. An optional `GITHUB_TOKEN` raises the GitHub API rate limit; it is sent only to GitHub, never to the user metrics endpoint. In Actions, the repository's built-in token is used. Set the repository variable `PAPERLEET_METRICS_URL` if the public website address changes. The first account snapshot requires the backend containing this endpoint to be deployed.

## 中文说明

- **Stars**：按当前仍在 Star 列表中的用户的 Star 日期累计；取消的 Star 无法从这个接口还原。
- **账号曲线**：按 UTC 创建日期累计，包含提前发放但尚未使用的内测账号，以及停用账号；不是活跃用户数或访问量。永久删除的账号不再计入。
- **Commit 曲线**：默认分支每个 UTC 自然周的非合并提交数，排除只更新图表的自动提交；本周数据尚未结束。

每日自动更新，图中注明数据时间。网站离线时保留上次成功的账号统计，不补造数据。GitHub 中仅保存日期和数量的汇总，完整用户数据库继续留在部署环境的私有数据卷中。
