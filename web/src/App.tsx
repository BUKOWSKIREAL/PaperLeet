import { useEffect, useRef, useState } from "react";
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
  Sun,
  Terminal,
  X,
  XCircle,
} from "lucide-react";
import { ArtifactCharts } from "./Visuals";
import Visuals from "./Lab";
import { isPassed, layers, status } from "./types";
import type { Category, Paper, Problem, Report } from "./types";
import { PaperDetail, PaperLibrary, paperModules } from "./PaperViews";
import { collectionIdFromRoute, paperCollections, papersInCollection } from "./paperCollections";
import CollectionDetail from "./CollectionDetail";
import { parseReport } from "./reports";
import { navigateToProblem, resolveAppLocation } from "./navigation";
import { LocaleProvider, useI18n } from "./i18n";
import Submissions from "./Submissions";
import Community, { UserProfile } from "./Community";
import Account from "./Account";
import { AccountNav, AccountProvider, useAccount } from "./social";
import { usePracticeWorkspace } from "./PracticeWorkspace";
import Brand from "./Brand";
import { getProblemStats, ProblemStats, useProblemStats } from "./ProblemStats";
import ResizableSplit from "./ResizableSplit";
import OnlinePresence from "./OnlinePresence";
import "./community.css";

const difficulty = (d: string) =>
  ({ E: "Easy", M: "Medium", H: "Hard" })[d] || d;
const editorFontSizes = [13, 14, 15, 16, 17, 18, 20];
type Theme = "light" | "dark";
function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem("paperleet.theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // Browser storage can be unavailable in private or embedded contexts.
  }
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  } catch {
    return "light";
  }
}
function initialEditorFontSize() {
  try {
    const saved = Number(localStorage.getItem("paperleet.editor.fontSize"));
    if (editorFontSizes.includes(saved)) return saved;
  } catch {
    // The editor remains usable when browser storage is unavailable.
  }
  return 15;
}
function getId() {
  const { route, canonicalUrl } = resolveAppLocation(location);
  if (canonicalUrl) history.replaceState(null, "", canonicalUrl);
  return route;
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
    <div className="highlight">
      <pre>
        <code>{code}</code>
      </pre>
    </div>
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
      const match = /\/problems\/([a-z_]+)\/([^/]+)\/README(?:\.en)?\.md$/.exec(
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
      <AccountProvider>
        <Workspace />
      </AccountProvider>
    </LocaleProvider>
  );
}
function Workspace() {
  const { locale, setLocale, t } = useI18n();
  const { user, health, healthError, refreshHealth, capabilities } =
    useAccount();
  const practice = usePracticeWorkspace();
  const { saved, update } = practice;
  const online = !!health?.judge.ready && !healthError;
  const hosted = health?.mode === "hosted";
  const testAuth = capabilities?.auth_mode === "test";
  const runController = useRef<AbortController | undefined>(undefined);
  const [problems, setProblems] = useState<Problem[]>([]),
    [categories, setCategories] = useState<Category[]>([]),
    [papers, setPapers] = useState<Paper[]>([]),
    [loadError, setLoadError] = useState(""),
    [id, setId] = useState(getId),
    [tab, setTab] = useState("description"),
    [rightTab, setRightTab] = useState("code"),
    [running, setRunning] = useState(false),
    [toast, setToast] = useState(""),
    [guide, setGuide] = useState(false),
    [font, setFont] = useState(initialEditorFontSize),
    [theme, setTheme] = useState<Theme>(initialTheme);
  const { stats: problemStats, refresh: refreshProblemStats, loading: statsLoading } = useProblemStats(id);
  const sourceProblem = problems.find(
    (p) => p.id === id || String(p.number) === id,
  );
  const localize = (p: Problem): Problem =>
    locale === "en" && p.translations?.en ? { ...p, ...p.translations.en } : p;
  const problem = sourceProblem ? localize(sourceProblem) : undefined;
  const submissionPage = id === "submit";
  const communityPage = id === "community" || id.startsWith("community/");
  const accountPage = id === "account";
  const profileUsername = id.startsWith("u/") ? id.slice(2) : "";
  const collectionId = collectionIdFromRoute(id);
  const selectedCollection = paperCollections.find((collection) => collection.id === collectionId);
  useEffect(() => {
    document.title = selectedCollection
      ? `${t(selectedCollection.name, selectedCollection.name_en)} - PaperLeet`
      : collectionId !== undefined ? `${t("合集未找到", "Collection not found")} - PaperLeet` : "PaperLeet";
  }, [collectionId, selectedCollection, locale]);
  const collectionModules = selectedCollection
    ? papersInCollection(papers, selectedCollection).flatMap((paper) => paperModules(paper, problems))
    : [];
  const current = problem ? saved[problem.id] || {} : {};
  const selectedPaper = papers.find((paper) =>
    problem ? paper.id === problem.paper_id : id === `papers/${paper.id}`,
  );
  const siblingModules = selectedPaper
    ? paperModules(selectedPaper, problems)
    : sourceProblem
      ? [sourceProblem]
      : [];
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
        const loadedPapers: Paper[] = Array.isArray(data.papers) ? [...data.papers] : [];
        // Shuffle on catalog load, keeping the order stable during browsing and filtering.
        for (let i = loadedPapers.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [loadedPapers[i], loadedPapers[j]] = [loadedPapers[j], loadedPapers[i]];
        }
        setPapers(loadedPapers);
        setCategories(Array.isArray(data.categories) ? data.categories : []);
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
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", h);
    return () => window.removeEventListener("hashchange", h);
  }, []);
  useEffect(() => {
    runController.current?.abort();
    setRunning(false);
    setToast("");
    setTab("description");
  }, [practice.owner]);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute(
      "content",
      theme === "dark" ? "#1a1a1a" : "#ffffff",
    );
  }, [theme]);
  function selectTheme(nextTheme: Theme) {
    setTheme(nextTheme);
    try {
      localStorage.setItem("paperleet.theme", nextTheme);
    } catch {
      // Theme selection still works for this session without browser storage.
    }
  }
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 6000);
    return () => clearTimeout(timer);
  }, [toast]);
  async function run() {
    if (
      !problem ||
      !practice.ready ||
      (hosted && (!user || !online || practice.conflicts[problem.id]))
    )
      return;
    const runOwner = practice.owner;
    const submittedCode = current.code ?? problem.starter;
    setRunning(true);
    try {
      if (hosted) await practice.flush();
      if (!practice.owns(runOwner)) return;
      const controller = new AbortController();
      runController.current = controller;
      const response = await fetch("/api/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problem: problem.id,
          code: submittedCode,
          ...(hosted ? { expected_user_id: user?.id } : {}),
        }),
        signal: AbortSignal.any([
          controller.signal,
          AbortSignal.timeout(180000),
        ]),
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
      void refreshProblemStats();
      if (!practice.owns(runOwner)) return;
      practice.applyReport(problem, report, runOwner);
      setTab("tests");
      setToast(t("判题已完成", "Run complete"));
    } catch (e) {
      if (!practice.owns(runOwner)) return;
      setToast(
        e instanceof Error && e.name !== "SyntaxError" && e.name !== "TypeError"
          ? e.message
          : t(
              "暂时无法连接判题服务，请稍后重试。你的代码已保存在浏览器。",
              "The judge is unavailable. Please try again shortly. Your code is saved in this browser.",
            ),
      );
    } finally {
      if (practice.owns(runOwner)) setRunning(false);
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
    /##\s*6[.、．]?\s*(?:怎么运行|编写与运行|Write and run|Running)[^\n]*\n[\s\S]*?(?=\n##\s*7|$)/,
    `## 6. ${t("编写与运行", "Write and run")}\n\n${hosted ? (testAuth ? t("使用内测账号登录后，可在右侧完成实现并运行测试。草稿与判题结果会私密同步到账号；全部通过后解锁坑点。", "Sign in with your test account, implement the module, and run tests. Drafts and judging results sync privately to your account; passing all tests unlocks Notes.") : t("受邀用户登录后，可在右侧完成实现并运行测试。草稿与判题结果会私密同步到账号；全部通过后解锁坑点。", "Invited members can sign in, implement the module, and run tests. Drafts and judging results sync privately to your account; passing all tests unlocks Notes.")) : t("在右侧编辑器中完成实现，点击「运行全部测试」。测试结果会在「测试」标签页显示；全部通过后解锁坑点。草稿会自动保存在当前浏览器。", "Complete the implementation in the editor and select Run all tests. Results appear in the Tests tab; passing all tests unlocks Notes. Drafts are saved automatically in this browser.")}\n`,
  );
  return (
    <div className={`app ${problem ? "workspace-view" : ""}`}>
      <header className="topbar">
        <div className="nav-inner">
          <Brand />
          <nav>
            <a
              className={
                !problem &&
                !submissionPage &&
                !communityPage &&
                !accountPage &&
                !profileUsername
                  ? "active"
                  : ""
              }
              href="/"
            >
              {t("论文题库", "Papers")}
            </a>
            <button
              className={`continue-link ${problem ? "active" : ""}`}
              disabled={!problem && (!practice.ready || practice.loading || !problems.length || (collectionId !== undefined && !selectedCollection))}
              onClick={() => {
                if (!problem && practice.ready && !practice.loading) {
                  const next = collectionId !== undefined
                    ? collectionModules.find((p) => status(p, saved[p.id]) !== "done") || collectionModules[0]
                    : selectedPaper
                    ? siblingModules.find((p) => !p.module.optional && status(p, saved[p.id]) !== "done") || siblingModules[0]
                    : problems.filter((p) => status(p, saved[p.id]) === "started")
                        .sort((a, b) => (saved[b.id]?.updated || "").localeCompare(saved[a.id]?.updated || ""))[0]
                      || problems[0];
                  if (next) navigateToProblem(next.number);
                }
              }}
            >
              {problem ? t("工作区", "Workspace") : t("继续练习", "Continue")}
            </button>
            <a className={submissionPage ? "active" : ""} href="/#/submit">
              {t("提交论文", "Submit paper")}
            </a>
            <a
              className={communityPage || profileUsername ? "active" : ""}
              href="/#/community"
            >
              {t("社区", "Community")}
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
            <OnlinePresence />
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
            <div
              className="theme-switcher"
              role="group"
              aria-label={t("外观", "Appearance")}
            >
              <button
                type="button"
                className={theme === "light" ? "selected" : ""}
                aria-label={t("浅色主题", "Light theme")}
                aria-pressed={theme === "light"}
                title={t("浅色主题", "Light theme")}
                onClick={() => selectTheme("light")}
              >
                <Sun size={15} aria-hidden="true" />
                <span className="theme-label">{t("浅色", "Light")}</span>
              </button>
              <button
                type="button"
                className={theme === "dark" ? "selected" : ""}
                aria-label={t("深色主题", "Dark theme")}
                aria-pressed={theme === "dark"}
                title={t("深色主题", "Dark theme")}
                onClick={() => selectTheme("dark")}
              >
                <Moon size={15} aria-hidden="true" />
                <span className="theme-label">{t("深色", "Dark")}</span>
              </button>
            </div>
            <AccountNav />
          </div>
        </div>
      </header>
      {communityPage ? (
        <Community
          postId={
            id === "community" ? undefined : id.slice("community/".length)
          }
        />
      ) : accountPage ? (
        <Account />
      ) : profileUsername ? (
        <UserProfile key={profileUsername} username={profileUsername} />
      ) : submissionPage ? (
        <Submissions
          key={`${health?.mode || "pending"}:${user?.id || "anonymous"}`}
          onPublished={loadCatalog}
        />
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
        selectedPaper ? (
          <PaperDetail
            paper={selectedPaper}
            problems={problems}
            saved={saved}
            stats={problemStats}
            onGuide={() => setGuide(true)}
          />
        ) : collectionId !== undefined ? (
          <CollectionDetail
            key={collectionId}
            collectionId={collectionId}
            progressReady={practice.ready && !practice.loading}
            papers={papers}
            problems={problems}
            saved={saved}
          />
        ) : (
          <PaperLibrary
            progressReady={practice.ready && !practice.loading}
            categories={categories}
            papers={papers}
            problems={problems}
            saved={saved}
            onGuide={() => setGuide(true)}
          />
        )
      ) : (
        <>
          <div className="workspace-toolbar">
            <a
              className="back-link"
              href={selectedPaper ? `#/papers/${selectedPaper.id}` : "#"}
            >
              <ArrowLeft size={15} />
              {selectedPaper
                ? `${selectedPaper.number}. ${selectedPaper.short_title}`
                : t("论文题库", "Papers")}
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
          <ResizableSplit
            as="main"
            className="workspace"
            axis="horizontal"
            storageKey="paperleet.layout.workspace.v1"
            initial={0.5}
            minFirst={320}
            minSecond={360}
            collapseAt={850}
            label={t("调整题面与代码区宽度", "Resize description and code panels")}
            resetLabel={t("拖动或使用方向键调整，双击或按 Enter 恢复默认", "Drag or use arrow keys to resize; double-click or press Enter to reset")}
            firstId="problem-description-panel"
            secondId="problem-editor-panel"
          >
            <section className="description-pane" id="problem-description-panel">
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
              <div className="description-scroll" key={problem.id}>
                {tab === "tests" ? (
                  <ReportView report={current.report} problem={problem} />
                ) : tab === "lab" ? (
                  <Visuals key={problem.id} problem={problem} report={current.report} />
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
                      <ProblemStats stats={getProblemStats(problemStats, problem.id)} loading={statsLoading} />
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
                    {problem.module && (
                      <section className="module-scope-card">
                        <h3>{t("本模块的目标", "Your implementation goal")}</h3>
                        <p>{t(problem.module.goal, problem.module.goal_en)}</p>
                        <p className="module-scope-detail">
                          {t(problem.module.scope, problem.module.scope_en)}
                        </p>
                        <div className="module-functions">
                          <span>{t("需要实现", "Implement")}</span>
                          {problem.module.required_functions.map((name) => (
                            <code key={name}>{name}</code>
                          ))}
                        </div>
                        {!!problem.module.provided_functions.length && (
                          <div className="module-functions provided-functions">
                            <span>{t("已提供", "Provided")}</span>
                            {problem.module.provided_functions.map((name) => (
                              <code key={name}>{name}</code>
                            ))}
                          </div>
                        )}
                      </section>
                    )}
                    {tab === "explanation" && (
                      <Visuals key={problem.id} problem={problem} report={current.report} />
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
                          <b>{problem.vizData?.kind === "toy_experiment" ? t("查看核心算法实验", "Explore the core experiment") : t("把公式放进实验台", "Explore the formulas")}</b>
                          <small>
                            {problem.vizData?.kind === "toy_experiment" ? t("对比参考实现与自己的运行结果", "Compare the reference with your run results") : t(
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
            <section className="editor-pane" id="problem-editor-panel">
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
                  <Visuals key={problem.id} problem={problem} report={current.report} />
                </div>
              ) : (
                <>
                  <ResizableSplit
                    className="editor-split"
                    axis="vertical"
                    storageKey="paperleet.layout.editor.v1"
                    initial={0.72}
                    minFirst={180}
                    minSecond={100}
                    label={t("调整代码与测试结果区高度", "Resize code and test results panels")}
                    resetLabel={t("拖动或使用方向键调整，双击或按 Enter 恢复默认", "Drag or use arrow keys to resize; double-click or press Enter to reset")}
                    firstId="problem-code-panel"
                    secondId="problem-results-panel"
                  >
                  <div className="editor-code-panel" id="problem-code-panel">
                  <div className="editor-tools">
                    <span>
                      <FileCode2 size={13} />
                      solution.py
                    </span>
                    <div>
                      <label className="editor-font-size">
                        <span aria-hidden="true">Aa</span>
                        <select
                          className="editor-font-size-select"
                          value={font}
                          title={t("代码字号", "Code font size")}
                          aria-label={t("代码字号", "Code font size")}
                          onChange={(event) => {
                            const size = Number(event.target.value);
                            setFont(size);
                            try {
                              localStorage.setItem(
                                "paperleet.editor.fontSize",
                                String(size),
                              );
                            } catch {
                              // Keep the selected size for this session.
                            }
                          }}
                        >
                          {editorFontSizes.map((size) => (
                            <option key={size} value={size}>{size} px</option>
                          ))}
                        </select>
                      </label>
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
                        disabled={!practice.ready}
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
                      key={`${practice.owner}:${problem.id}`}
                      readOnly={!practice.ready}
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
                      {practice.loading
                        ? t("正在加载工作区…", "Loading workspace…")
                        : hosted
                          ? !user
                            ? t(
                                "草稿仅保存在此浏览器，登录后使用独立工作区",
                                "Browser-only draft · sign in for your account workspace",
                              )
                            : practice.error
                              ? t(
                                  "同步失败，草稿保留在当前浏览器",
                                  "Sync failed · draft kept in this browser",
                                )
                              : practice.conflicts[problem.id]
                                ? t(
                                    "草稿版本冲突，请选择保留的版本",
                                    "Draft conflict · choose a version below",
                                  )
                                : practice.saving || practice.pending
                                  ? t("正在同步草稿…", "Syncing draft…")
                                  : t("已同步到账号", "Synced to your account")
                          : current.code !== undefined
                            ? t("草稿已自动保存", "Draft saved automatically")
                            : t("初始代码", "Starter code")}
                    </span>
                    <span>
                      UTF-8 <i>·</i> Python
                    </span>
                  </div>
                  </div>
                  <div className="editor-results-panel" id="problem-results-panel">
                  {(practice.error ||
                    practice.conflicts[problem.id] ||
                    (!health && healthError) ||
                    (hosted && (!user || !online))) && (
                    <div className="practice-notice" role="status">
                      {practice.conflicts[problem.id] ? (
                        <>
                          <span>
                            {t(
                              "其他设备更新了这道题的草稿。选择保留哪个版本；当前代码尚未覆盖服务器。",
                              "Another device updated this draft. Choose which version to keep; your current code has not overwritten the server.",
                            )}
                          </span>
                          <button
                            onClick={() =>
                              practice.resolveConflict(problem.id, false)
                            }
                          >
                            {t("使用服务器版本", "Use server draft")}
                          </button>
                          <button
                            onClick={() =>
                              practice.resolveConflict(problem.id, true)
                            }
                          >
                            {t("保留当前版本", "Keep this draft")}
                          </button>
                        </>
                      ) : practice.error ? (
                        <>
                          <span>{practice.error}</span>
                          <button onClick={practice.retry}>
                            {t("重试同步", "Retry sync")}
                          </button>
                        </>
                      ) : !health ? (
                        <>
                          <span>
                            {t(
                              "无法确认服务配置，工作区尚未加载。",
                              "Service configuration is unavailable. Your workspace has not been loaded.",
                            )}
                          </span>
                          <button onClick={() => void refreshHealth()}>
                            {t("重试", "Retry")}
                          </button>
                        </>
                      ) : !user ? (
                        <>
                          <span>
                            {testAuth
                              ? t(
                                  "使用管理员提供的内测账号和访问码登录后，即可运行测试。",
                                  "Sign in with your administrator-issued test account and access code to run tests.",
                                )
                              : t(
                                  "在线内测仅向受邀且已验证机构邮箱的用户开放判题。",
                                  "Online judging is available to invited members with a verified institutional email.",
                                )}
                          </span>
                          <a href="#/account">
                            {t("登录以运行", "Sign in to run")}
                          </a>
                        </>
                      ) : (
                        <>
                          <span>
                            {t(
                              "判题沙箱尚未就绪，请稍后重试。草稿仍会同步。",
                              "The judge sandbox is not ready. Please try again shortly; your drafts will still sync.",
                            )}
                          </span>
                          <button onClick={() => void refreshHealth()}>
                            {t("刷新状态", "Refresh status")}
                          </button>
                        </>
                      )}
                    </div>
                  )}
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
                  </div>
                  </ResizableSplit>
                  <div className="editor-bottom">
                    <span>
                      {running ? (
                        <>
                          <span className="spinner" />
                          {t("正在判题…", "Running tests…")}
                        </>
                      ) : hosted && !user ? (
                        <a href="#/account">
                          {t("登录后运行", "Sign in to run")}
                        </a>
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
                      disabled={
                        running ||
                        !practice.ready ||
                        (hosted &&
                          (!user ||
                            !online ||
                            !!practice.conflicts[problem.id]))
                      }
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
          </ResizableSplit>
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
