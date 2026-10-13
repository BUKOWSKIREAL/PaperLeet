import { useEffect, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import {
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Code2,
  Copy,
  ExternalLink,
  FileCode2,
  FlaskConical,
  Globe2,
  LockKeyhole,
  Moon,
  Play,
  RotateCcw,
  Settings2,
  Sun,
  Terminal,
  X,
  XCircle,
} from "lucide-react";
import { ArtifactCharts } from "./Visuals";
import Visuals from "./Lab";
import { isPassed, layers, status } from "./types";
import type { Paper, Problem, Report, Saved } from "./types";
import { PaperDetail, PaperLibrary, paperModules } from "./PaperViews";
import { decodeProblemHash, parseReport, parseSaved } from "./reports";
import { LocaleProvider, useI18n } from "./i18n";
import Submissions from "./Submissions";

const STORAGE = "paperleet.workspace.v1";
const difficulty = (d: string) =>
  ({ E: "Easy", M: "Medium", H: "Hard" })[d] || d;
function loadSaved(): Record<string, Saved> {
  try {
    return parseSaved(JSON.parse(localStorage.getItem(STORAGE) || "{}"));
  } catch {
    return {};
  }
}
function getId() {
  return decodeProblemHash(location.hash);
}
function download(name: string, content: string) {
  const url = URL.createObjectURL(
    new Blob([content], { type: "text/plain;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
export function Highlight({
  code,
  lang = "python",
}: {
  code: string;
  lang?: string;
}) {
  const [html, setHtml] = useState("");
  useEffect(() => {
    let active = true;
    import("./highlight")
      .then(({ highlight }) => highlight(code, lang))
      .then((h) => {
        if (active) setHtml(h);
      })
      .catch(() => {
        if (active) setHtml("");
      });
    return () => {
      active = false;
    };
  }, [code, lang]);
  return html ? (
    <div className="highlight" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <pre>
      <code>{code}</code>
    </pre>
  );
}
export function Prose({
  content,
  problemId,
  onViz,
}: {
  content: string;
  problemId?: string;
  onViz?: () => void;
}) {
  const { t } = useI18n();
  const processed = content.replace(
    /<viz\s+id=["']([^"']+)["']\s*\/>/g,
    `[${t("打开交互实验", "Open interactive lab")}](#viz-$1)`,
  );
  function Link({
    href,
    children,
  }: {
    href?: string;
    children?: React.ReactNode;
  }) {
    if (href?.startsWith("#viz-"))
      return (
        <button className="text-button inline-lab-link" onClick={onViz}>
          <FlaskConical size={13} />
          {children}
        </button>
      );
    if (href && /^https?:/.test(href))
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    if (href?.includes("README.md") && problemId) {
      const resolved = new URL(
        href,
        `https://local/problems/${problemId}/README.md`,
      ).pathname;
      const match = /\/problems\/(rl|diffusion)\/([^/]+)\/README.md$/.exec(
        resolved,
      );
      if (match) return <a href={`#/${match[1]}/${match[2]}`}>{children}</a>;
    }
    return <a href={href}>{children}</a>;
  }
  return (
    <div className="prose">
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a: Link,
          pre: ({ children }) => <>{children}</>,
          code: ({ className, children, ...props }) => {
            const lang = /language-(\w+)/.exec(className || "")?.[1];
            return lang ? (
              <Highlight
                code={String(children).replace(/\n$/, "")}
                lang={lang}
              />
            ) : (
              <code {...props} className={className}>
                {children}
              </code>
            );
          },
        }}
      >
        {processed}
      </Markdown>
    </div>
  );
}
function Brand() {
  const { t } = useI18n();
  return (
    <a
      href="#"
      className="brand"
      aria-label={t("PaperLeet 首页", "PaperLeet home")}
    >
      <span className="brand-mark">
        <Code2 size={25} />
      </span>
      <span>
        Paper<span>Leet</span>
      </span>
      <b>BETA</b>
    </a>
  );
}
function ReportView({
  report,
  problem,
}: {
  report?: Report;
  problem: Problem;
}) {
  const { t, locale } = useI18n();
  const labels = t(
    "接口契约|参考对齐|数学性质|真实行为",
    "Contract|Reference|Properties|Behavior",
  ).split("|");
  if (!report)
    return (
      <div className="empty-state">
        <div className="empty-icon">
          <FlaskConical size={28} />
        </div>
        <h3>{t("用测试验证你的实现", "Validate your implementation")}</h3>
        <p>
          {t(
            "在右侧编写代码，然后点击「运行全部测试」。",
            "Write your code on the right, then select Run all tests.",
          )}
          <br />
          {t(
            "测试结果和失败原因会直接显示在这里。",
            "Results and failure details will appear here.",
          )}
        </p>
      </div>
    );
  const passed = report.tests.filter((test) => test.passed).length,
    total = report.tests.length,
    allPassed = !report.error && total > 0 && passed === total;
  return (
    <div className="report-view">
      <div className={`result-head ${allPassed ? "success" : "failure"}`}>
        <div>
          {allPassed ? <CheckCircle2 size={24} /> : <XCircle size={24} />}
          <h3>
            {report.error
              ? t("判题未完成", "Run incomplete")
              : isPassed(problem, report)
                ? "Accepted"
                : allPassed
                  ? t("所选测试通过", "Selected tests passed")
                  : t("需要再检查一下", "Tests need attention")}
          </h3>
        </div>
        <span>
          {passed} / {total} {t("项通过", "tests passed")}
        </span>
      </div>
      {report.error && <pre className="failure-message">{report.error}</pre>}
      <div className="report-meta">
        <span>{t("最近一次运行", "Latest run")}</span>
        <span>
          {report.created_at
            ? new Date(report.created_at).toLocaleString(
                locale === "zh" ? "zh-CN" : "en-US",
              )
            : ""}
        </span>
      </div>
      <div className="test-layers">
        {layers.map((l, i) => {
          const [a, b] = report.summary[l] || [0, 0];
          return (
            <div key={l}>
              <div>
                <b>{l}</b>
                <span>
                  {a}/{b}
                </span>
              </div>
              <small>{labels[i]}</small>
              <div className="mini-progress">
                <i style={{ width: `${b ? (a / b) * 100 : 0}%` }} />
              </div>
            </div>
          );
        })}
      </div>
      {report.tests.map((test, i) => (
        <details
          key={`${test.nodeid || test.id}-${i}`}
          className={`test-detail ${test.passed ? "passed" : "failed"}`}
          open={!test.passed}
        >
          <summary>
            {test.passed ? <CheckCircle2 size={15} /> : <XCircle size={15} />}
            <span>{test.id}</span>
            <b>{test.layer}</b>
            <small>{(test.duration_s || 0).toFixed(2)}s</small>
          </summary>
          <div className="test-body">
            <p className="test-why">
              <b>{t("为什么检查这个性质", "Why this property matters")}</b>
              {test.why ||
                t("见本题测试说明。", "See the problem's test specification.")}
            </p>
            {test.message && (
              <pre className="failure-message">{test.message}</pre>
            )}
            {test.artifacts && Object.keys(test.artifacts).length > 0 && (
              <ArtifactCharts data={test.artifacts} />
            )}
          </div>
        </details>
      ))}
      {report.mutation && (
        <section className="mutation">
          <h4>{t("错误变体检查", "Mutation checks")}</h4>
          {Object.entries(report.mutation).map(([id, caught]) => (
            <div key={id}>
              <b>
                {caught.length ? "✓" : "×"} {id}
              </b>
              <span>
                {caught.length
                  ? caught.join(", ")
                  : t("当前测试未捕获", "Not caught by current tests")}
              </span>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
export default function App() {
  return (
    <LocaleProvider>
      <Workspace />
    </LocaleProvider>
  );
}
function Workspace() {
  const { locale, setLocale, t } = useI18n();
  const [problems, setProblems] = useState<Problem[]>([]),
    [papers, setPapers] = useState<Paper[]>([]),
    [loadError, setLoadError] = useState(""),
    [id, setId] = useState(getId),
    [saved, setSaved] = useState(loadSaved),
    [tab, setTab] = useState("description"),
    [rightTab, setRightTab] = useState("code"),
    [online, setOnline] = useState(false),
    [running, setRunning] = useState(false),
    [toast, setToast] = useState(""),
    [guide, setGuide] = useState(false),
    [font, setFont] = useState(13),
    [theme, setTheme] = useState(
      () => localStorage.getItem("paperleet.theme") || "dark",
    );
  const sourceProblem = problems.find(
    (p) => p.id === id || String(p.number) === id,
  );
  const localize = (p: Problem): Problem =>
    locale === "en" && p.translations?.en ? { ...p, ...p.translations.en } : p;
  const problem = sourceProblem ? localize(sourceProblem) : undefined;
  const submissionPage = id === "submit";
  const current = problem ? saved[problem.id] || {} : {};
  const selectedPaper = papers.find((paper) =>
    problem ? paper.id === problem.paper_id : id === `papers/${paper.id}`,
  );
  const siblingModules = selectedPaper ? paperModules(selectedPaper, problems) : sourceProblem ? [sourceProblem] : [];
  const moduleIndex = siblingModules.findIndex((p) => p.id === problem?.id);
  function loadCatalog() {
    return fetch(
      `${import.meta.env.BASE_URL}data/catalog.json?v=${Date.now()}`,
      { cache: "no-store" },
    )
      .then(async (r) => {
        if (!r.ok)
          throw Error(
            t(
              "暂时无法加载题库，请稍后刷新。",
              "The problem catalog is unavailable. Please refresh shortly.",
            ),
          );
        return r.json();
      })
      .then((data) => {
        setProblems(Array.isArray(data) ? data : data.problems);
        setPapers(Array.isArray(data.papers) ? data.papers : []);
        setLoadError("");
      })
      .catch((e) => setLoadError(e.message));
  }
  useEffect(() => {
    void loadCatalog();
    const h = () => {
      setId(getId());
      setTab("description");
      setRightTab("code");
    };
    window.addEventListener("hashchange", h);
    return () => window.removeEventListener("hashchange", h);
  }, []);
  useEffect(() => {
    let live = true;
    const ping = () =>
      fetch("/api/health", { signal: AbortSignal.timeout(2000) })
        .then((r) => r.json())
        .then((r) => {
          if (live) setOnline(r.ok === true || r.status === "ok");
        })
        .catch(() => {
          if (live) setOnline(false);
        });
    void ping();
    const timer = setInterval(ping, 15000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("paperleet.theme", theme);
  }, [theme]);
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE, JSON.stringify(saved));
    } catch {
      setToast(
        t(
          "浏览器存储空间不足，请下载代码备份。",
          "Browser storage is full. Download your code to keep a backup.",
        ),
      );
    }
  }, [saved]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 6000);
    return () => clearTimeout(timer);
  }, [toast]);
  function update(pid: string, change: Partial<Saved>) {
    setSaved((prev) => ({
      ...prev,
      [pid]: { ...prev[pid], ...change, updated: new Date().toISOString() },
    }));
  }
  async function run() {
    if (!problem) return;
    setRunning(true);
    try {
      const response = await fetch("/api/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problem: problem.id,
          code: current.code ?? problem.starter,
        }),
        signal: AbortSignal.timeout(180000),
      });
      const data = await response.json();
      if (
        !response.ok ||
        (data.error && !Array.isArray(data.tests) && !data.report)
      )
        throw Error(
          data.error || t("判题服务返回错误", "The judge returned an error"),
        );
      const report = parseReport(data.report || data);
      if (report.problem !== problem.id)
        throw Error(
          t("判题服务返回了无效结果", "The judge returned an invalid result"),
        );
      update(problem.id, {
        report,
        acceptedReport: isPassed(problem, report)
          ? report
          : isPassed(problem, current.acceptedReport)
            ? current.acceptedReport
            : isPassed(problem, current.report) ? current.report : undefined,
      });
      setTab("tests");
      setOnline(true);
      setToast(t("判题已完成", "Run complete"));
    } catch (e) {
      setToast(
        e instanceof Error && e.name !== "SyntaxError" && e.name !== "TypeError"
          ? e.message
          : t(
              "暂时无法连接判题服务，请稍后重试。你的代码已保存在浏览器。",
              "The judge is unavailable. Please try again shortly. Your code is saved in this browser.",
            ),
      );
    } finally {
      setRunning(false);
    }
  }
  async function copy(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setToast(t("已复制", "Copied"));
    } catch {
      setToast(
        t(
          "剪贴板不可用，请手动复制。",
          "Clipboard unavailable. Please copy manually.",
        ),
      );
    }
  }
  const tabs = [
    ["description", t("题目", "Description")],
    ["explanation", t("讲解", "Explanation")],
    ["lab", t("实验", "Lab")],
    ["tests", t("测试", "Tests")],
    ["notes", t("坑点", "Notes")],
    ["solution", t("参考解", "Solution")],
  ];
  const webReadme = problem?.readme.replace(
    /##\s*6[.、．]?\s*怎么运行[^\n]*\n[\s\S]*?(?=\n##\s*7|$)/,
    `## 6. ${t("编写与运行", "Write and run")}\n\n${t("在右侧编辑器中完成实现，点击「运行全部测试」。测试结果会在「测试」标签页显示；全部通过后解锁坑点。草稿会自动保存在当前浏览器。", "Complete the implementation in the editor and select Run all tests. Results appear in the Tests tab; passing all tests unlocks Notes. Drafts are saved automatically in this browser.")}\n`,
  );
  return (
    <div className={`app ${problem ? "workspace-view" : ""}`}>
      <header className="topbar">
        <div className="nav-inner">
          <Brand />
          <nav>
            <a className={!problem && !submissionPage ? "active" : ""} href="#">
              {t("论文题库", "Papers")}
            </a>
            <button
              className={problem ? "active" : ""}
              onClick={() => {
                if (!problem) {
                  const next =
                    problems.find(
                      (p) => status(p, saved[p.id]) === "started",
                    ) || problems[0];
                  if (next) location.hash = `/${next.number}`;
                }
              }}
            >
              {problem ? t("工作区", "Workspace") : t("继续练习", "Continue")}
            </button>
            <a className={submissionPage ? "active" : ""} href="#/submit">
              {t("提交论文", "Submit paper")}
            </a>
            <button onClick={() => setGuide(true)}>
              {t("使用指南", "Guide")}
            </button>
          </nav>
          <div className="nav-actions">
            <span
              className={`connection ${online ? "connected" : ""}`}
              title={t(
                "网页直接提交代码并显示测试结果",
                "Run code and view test results in this workspace",
              )}
            >
              <i />
              {online
                ? t("判题服务已连接", "Judge connected")
                : t("判题服务未连接", "Judge offline")}
            </span>
            <button
              className="language-toggle"
              aria-label={t("切换到英文", "Switch to Chinese")}
              onClick={() => setLocale(locale === "zh" ? "en" : "zh")}
            >
              <Globe2 size={15} />
              <span className={locale === "zh" ? "selected" : ""}>中</span>
              <i>/</i>
              <span className={locale === "en" ? "selected" : ""}>EN</span>
            </button>
            <button
              className="icon-button"
              aria-label={t("切换明暗主题", "Toggle color theme")}
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </button>
          </div>
        </div>
      </header>
      {submissionPage ? (
        <Submissions onPublished={loadCatalog} />
      ) : loadError ? (
        <main className="load-error">
          <Terminal size={35} />
          <h2>{t("题库暂时不可用", "Catalog unavailable")}</h2>
          <p>{loadError}</p>
          <button className="button" onClick={loadCatalog}>
            {t("重新加载", "Reload")}
          </button>
        </main>
      ) : !problems.length ? (
        <main className="loading">
          <span className="spinner" />
          {t("正在加载论文题库…", "Loading papers…")}
        </main>
      ) : !problem ? (
        selectedPaper ? <PaperDetail paper={selectedPaper} problems={problems} saved={saved} onGuide={() => setGuide(true)} /> :
        <PaperLibrary papers={papers} problems={problems} saved={saved} onGuide={() => setGuide(true)} />
      ) : (
        <>
          <div className="workspace-toolbar">
            <a className="back-link" href={selectedPaper ? `#/papers/${selectedPaper.id}` : "#"}>
              <ArrowLeft size={15} />
              {selectedPaper ? `${selectedPaper.number}. ${selectedPaper.short_title}` : t("论文题库", "Papers")}
            </a>
            <span className="toolbar-divider" />
            <b>
              {problem.number}. {problem.title}
            </b>
            <div className="problem-navigation">
              <button
                className="icon-button"
                aria-label={t("上一个模块", "Previous module")}
                disabled={moduleIndex <= 0}
                onClick={() =>
                  (location.hash = `/${siblingModules[moduleIndex - 1].number}`)
                }
              >
                <ChevronLeft size={15} />
              </button>
              <span>
                {moduleIndex + 1} / {siblingModules.length}
              </span>
              <button
                className="icon-button"
                aria-label={t("下一个模块", "Next module")}
                disabled={moduleIndex >= siblingModules.length - 1}
                onClick={() =>
                  (location.hash = `/${siblingModules[moduleIndex + 1].number}`)
                }
              >
                <ChevronRight size={15} />
              </button>
            </div>
          </div>
          <main className="workspace">
            <section className="description-pane">
              <div className="pane-tabs">
                {tabs.map(([key, label]) => (
                  <button
                    key={key}
                    className={tab === key ? "selected" : ""}
                    onClick={() => setTab(key)}
                  >
                    {key === "description" ? (
                      <FileCode2 size={14} />
                    ) : key === "lab" ? (
                      <FlaskConical size={14} />
                    ) : key === "notes" &&
                      status(problem, current) !== "done" ? (
                      <LockKeyhole size={12} />
                    ) : null}
                    {label}
                    {key === "tests" && current.report && (
                      <i
                        className={
                          isPassed(problem, current.report)
                            ? "green-dot"
                            : "yellow-dot"
                        }
                      />
                    )}
                  </button>
                ))}
              </div>
              <div className="description-scroll">
                {tab === "tests" ? (
                  <ReportView report={current.report} problem={problem} />
                ) : tab === "lab" ? (
                  <Visuals key={problem.id} problem={problem} />
                ) : tab === "notes" ? (
                  status(problem, current) !== "done" ? (
                    <div className="empty-state">
                      <div className="empty-icon">
                        <LockKeyhole size={25} />
                      </div>
                      <h3>
                        {t(
                          "通过测试后解锁坑点",
                          "Pass the tests to unlock notes",
                        )}
                      </h3>
                      <p>
                        {t(
                          "完成全部测试后，查看论文与常见实现之间的细节差异。",
                          "Complete every test to explore subtle differences between the paper and common implementations.",
                        )}
                      </p>
                      <button
                        className="button"
                        onClick={() => setTab("tests")}
                      >
                        {t("查看测试进度", "View test progress")}{" "}
                        <ArrowRight size={15} />
                      </button>
                    </div>
                  ) : (
                    <Prose
                      content={problem.notes}
                      problemId={problem.id}
                      onViz={() => setTab("lab")}
                    />
                  )
                ) : tab === "solution" ? (
                  <section className="solution-panel">
                    <h2>{t("参考实现", "Reference implementation")}</h2>
                    <p>
                      {t(
                        "建议先用测试结果定位问题，再对照参考解。",
                        "Use test results to find the issue before checking the reference.",
                      )}
                    </p>
                    <details>
                      <summary>
                        <BookOpen size={16} />
                        {t(
                          "展开参考解（包含答案）",
                          "Reveal reference solution (spoilers)",
                        )}
                      </summary>
                      {problem.solution ? (
                        <Highlight code={problem.solution} />
                      ) : (
                        <div className="notice">
                          {t(
                            "参考实现尚未提供。",
                            "No reference implementation is available yet.",
                          )}
                        </div>
                      )}
                    </details>
                  </section>
                ) : (
                  <>
                    <div className="problem-heading">
                      <div className="heading-kicker">
                        <FileCode2 size={18} />
                        <span>
                          {t("模块", "MODULE")} {problem.number}
                        </span>
                        {status(problem, current) === "done" && (
                          <span className="accepted">
                            <CheckCircle2 size={14} />
                            {t("已通过", "Solved")}
                          </span>
                        )}
                      </div>
                      <h1>
                        {problem.number}. {problem.title}
                      </h1>
                      <div className="heading-tags">
                        <span className={`difficulty ${problem.difficulty}`}>
                          {difficulty(problem.difficulty)}
                        </span>
                        <span>PyTorch</span>
                        <span>{problem.estimated_minutes || 25} min</span>
                        {problem.parent && (
                          <a
                            href={`#/${problems.find((p) => p.id === problem.parent)?.number || problem.parent}`}
                          >
                            {t("前置题", "Prerequisite")}:{" "}
                            {problems.find((p) => p.id === problem.parent)
                              ?.number || "—"}
                          </a>
                        )}
                      </div>
                    </div>
                    <a
                      className="paper-card"
                      href={
                        problem.paper?.url ||
                        `https://arxiv.org/abs/${problem.paper?.arxiv}`
                      }
                      target="_blank"
                      rel="noreferrer"
                    >
                      <BookOpen size={19} />
                      <div>
                        <small>
                          {t("阅读论文", "READ THE PAPER")}{" "}
                          <span>{problem.paper?.venue}</span>
                        </small>
                        <strong>{problem.paper?.title}</strong>
                        <p>
                          {Array.isArray(problem.paper?.authors)
                            ? problem.paper.authors.join(", ")
                            : problem.paper?.authors}
                        </p>
                      </div>
                      <ExternalLink size={14} />
                    </a>
                    {problem.module && <section className="module-scope-card">
                      <h3>{t("本模块的目标", "Your implementation goal")}</h3>
                      <p>{t(problem.module.goal, problem.module.goal_en)}</p>
                      <p className="module-scope-detail">{t(problem.module.scope, problem.module.scope_en)}</p>
                      <div className="module-functions"><span>{t("需要实现", "Implement")}</span>{problem.module.required_functions.map((name) => <code key={name}>{name}</code>)}</div>
                      {!!problem.module.provided_functions.length && <div className="module-functions provided-functions"><span>{t("已提供", "Provided")}</span>{problem.module.provided_functions.map((name) => <code key={name}>{name}</code>)}</div>}
                    </section>}
                    {tab === "explanation" && (
                      <Visuals key={problem.id} problem={problem} />
                    )}
                    {locale === "en" && !problem.translations?.en && (
                      <div className="notice">
                        English problem content is not available yet. The
                        original text appears below.
                      </div>
                    )}
                    <Prose
                      content={webReadme || ""}
                      problemId={problem.id}
                      onViz={() => setTab("lab")}
                    />
                    {tab === "description" && (
                      <button
                        className="lab-callout"
                        onClick={() => setTab("lab")}
                      >
                        <FlaskConical size={21} />
                        <span>
                          <b>{t("把公式放进实验台", "Explore the formulas")}</b>
                          <small>
                            {t(
                              "调整参数，观察结果怎样变化",
                              "Adjust parameters and see what changes",
                            )}
                          </small>
                        </span>
                        <ArrowRight size={18} />
                      </button>
                    )}
                  </>
                )}
              </div>
            </section>
            <section className="editor-pane">
              <div className="pane-tabs editor-tabs">
                <button
                  className={rightTab === "code" ? "selected" : ""}
                  onClick={() => setRightTab("code")}
                >
                  <Code2 size={15} />
                  {t("代码", "Code")}
                </button>
                <button
                  className={rightTab === "lab" ? "selected" : ""}
                  onClick={() => setRightTab("lab")}
                >
                  <FlaskConical size={14} />
                  {t("实验台", "Lab")}
                </button>
                <span className="editor-language">Python 3 · PyTorch</span>
              </div>
              {rightTab === "lab" ? (
                <div className="editor-lab">
                  <Visuals key={problem.id} problem={problem} />
                </div>
              ) : (
                <>
                  <div className="editor-tools">
                    <span>
                      <FileCode2 size={13} />
                      solution.py
                    </span>
                    <div>
                      <button
                        className="icon-button"
                        title={t("切换字号", "Change font size")}
                        aria-label={t("切换代码字号", "Change code font size")}
                        onClick={() =>
                          setFont(font === 13 ? 15 : font === 15 ? 17 : 13)
                        }
                      >
                        <Settings2 size={14} />
                      </button>
                      <button
                        className="icon-button"
                        title={t("复制代码", "Copy code")}
                        aria-label={t("复制代码", "Copy code")}
                        onClick={() => copy(current.code ?? problem.starter)}
                      >
                        <Copy size={14} />
                      </button>
                      <button
                        className="icon-button"
                        title={t("下载实现", "Download code")}
                        aria-label={t("下载实现", "Download code")}
                        onClick={() =>
                          download(
                            `problem-${problem.number}.py`,
                            current.code ?? problem.starter,
                          )
                        }
                      >
                        <ArrowDownToLine size={14} />
                      </button>
                      <button
                        className="icon-button"
                        title={t(
                          "恢复初始代码（先下载备份）",
                          "Reset code (download backup first)",
                        )}
                        aria-label={t(
                          "恢复初始代码并备份",
                          "Reset starter with backup",
                        )}
                        onClick={() => {
                          const hasDraft =
                            !!current.code && current.code !== problem.starter;
                          if (hasDraft)
                            download(
                              `problem-${problem.number}-backup.py`,
                              current.code!,
                            );
                          update(problem.id, { code: problem.starter });
                          setToast(
                            hasDraft
                              ? t(
                                  "已恢复初始代码，原代码已下载备份。",
                                  "Starter restored. Your previous code was downloaded as a backup.",
                                )
                              : t("已恢复初始代码。", "Starter restored."),
                          );
                        }}
                      >
                        <RotateCcw size={14} />
                      </button>
                    </div>
                  </div>
                  <div className="code-editor">
                    <CodeMirror
                      key={problem.id}
                      value={current.code ?? problem.starter}
                      height="100%"
                      theme={theme === "dark" ? "dark" : "light"}
                      extensions={[python()]}
                      onChange={(value) => update(problem.id, { code: value })}
                      basicSetup={{
                        lineNumbers: true,
                        foldGutter: true,
                        highlightActiveLine: true,
                        autocompletion: true,
                      }}
                      style={{ fontSize: font }}
                      aria-label={`${problem.title} ${t("Python 代码编辑器", "Python code editor")}`}
                    />
                  </div>
                  <div className="editor-status">
                    <span>
                      <Check size={12} />
                      {current.code !== undefined
                        ? t("草稿已自动保存", "Draft saved automatically")
                        : t("初始代码", "Starter code")}
                    </span>
                    <span>
                      UTF-8 <i>·</i> Python
                    </span>
                  </div>
                  <div className="console">
                    <div className="console-header">
                      <span>
                        <Terminal size={14} />
                        {t("测试结果", "Test results")}
                      </span>
                      {current.report && (
                        <button
                          className="text-button"
                          onClick={() => setTab("tests")}
                        >
                          {t("查看详情", "View details")}{" "}
                          <ArrowRight size={13} />
                        </button>
                      )}
                    </div>
                    <p>
                      {running
                        ? t(
                            "正在执行测试，结果将自动显示。",
                            "Running tests. Results will appear automatically.",
                          )
                        : current.report
                          ? `${current.report.tests.filter((test) => test.passed).length} / ${current.report.tests.length} ${t("项测试通过。修改代码后可再次运行。", "tests passed. Edit your code and run again.")}`
                          : t(
                              "完成实现后，点击下方按钮运行测试。",
                              "Complete the implementation, then run the tests below.",
                            )}
                    </p>
                  </div>
                  <div className="editor-bottom">
                    <span>
                      {running ? (
                        <>
                          <span className="spinner" />
                          {t("正在判题…", "Running tests…")}
                        </>
                      ) : online ? (
                        <>
                          <i className="green-dot" />
                          {t("可以运行", "Ready")}
                        </>
                      ) : (
                        <>
                          <i className="yellow-dot" />
                          {t("判题服务暂时不可用", "Judge unavailable")}
                        </>
                      )}
                    </span>
                    <button
                      className="button run-button"
                      disabled={running}
                      onClick={run}
                    >
                      <Play size={15} fill="currentColor" />
                      {running
                        ? t("运行中", "Running")
                        : t("运行全部测试", "Run all tests")}
                    </button>
                  </div>
                </>
              )}
            </section>
          </main>
        </>
      )}
      {guide && (
        <div className="modal-backdrop" onClick={() => setGuide(false)}>
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="guide-title"
            className="guide-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="icon-button modal-close"
              aria-label={t("关闭指南", "Close guide")}
              onClick={() => setGuide(false)}
            >
              <X size={19} />
            </button>
            <div className="eyebrow">
              <BookOpen size={16} />
              PAPERLEET
            </div>
            <h2 id="guide-title">
              {t("实现论文里的核心模块", "Implement a paper’s core modules")}
            </h2>
            <p>
              {t(
                "阅读论文、编写代码、运行测试都在这个工作区完成。",
                "Read the paper, write code, and run tests in this workspace.",
              )}
            </p>
            <ol>
              <li>
                <b>{t("选择论文与模块", "Choose a paper and module")}</b>
                <small>
                  {t(
                    "先了解论文的核心思想和练习范围，再选择模块，阅读接口与约定。",
                    "Read the paper’s core idea and selected scope, then choose a module and review its contract.",
                  )}
                </small>
              </li>
              <li>
                <b>{t("在编辑器中实现", "Implement in the editor")}</b>
                <small>
                  {t(
                    "草稿自动保存；实验台可以帮助你观察公式的行为。",
                    "Drafts are saved automatically. Use the lab to explore how the formulas behave.",
                  )}
                </small>
              </li>
              <li>
                <b>{t("运行全部测试", "Run all tests")}</b>
                <small>
                  {t(
                    "检查失败原因并修改代码。全部通过后，解锁这道题的坑点。",
                    "Read failure details and revise your code. Passing all tests unlocks the problem's notes.",
                  )}
                </small>
              </li>
              <li>
                <b>{t("提交新论文", "Contribute a paper")}</b>
                <small>
                  {t(
                    "提交论文和官方代码来源，生成候选实现与测试；通过验证和人工评审后才会加入题库。",
                    "Submit a paper and official code source. Generated code and tests enter the catalog only after validation and human review.",
                  )}
                </small>
              </li>
            </ol>
            <button className="button" onClick={() => setGuide(false)}>
              {t("开始练习", "Start practicing")} <ArrowRight size={15} />
            </button>
          </section>
        </div>
      )}
      {toast && (
        <div role="status" className="toast">
          <CheckCircle2 size={16} />
          {toast}
          <button
            className="icon-button"
            aria-label={t("关闭提示", "Dismiss notification")}
            onClick={() => setToast("")}
          >
            <X size={13} />
          </button>
        </div>
      )}
    </div>
  );
}
