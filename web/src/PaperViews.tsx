import { useState } from "react";
import {
  ArrowLeft, ArrowRight, BookOpen, CheckCircle2, ChevronDown,
  ChevronRight, Circle, ExternalLink, FileCode2, Layers3, ListFilter, Plus, Search,
} from "lucide-react";
import { useI18n } from "./i18n";
import { layers, status } from "./types";
import type { Category, Paper, Problem, Saved } from "./types";

type Practice = { problems: Problem[]; saved: Record<string, Saved> };
export function paperModules(paper: Paper, problems: Problem[]) {
  return paper.module_ids.flatMap((id) => {
    const problem = problems.find((p) => p.id === id);
    return problem ? [problem] : [];
  });
}
export function paperProgress(paper: Paper, { problems, saved }: Practice) {
  const modules = paperModules(paper, problems).filter((p) => !p.module.optional);
  const completed = modules.filter((p) => status(p, saved[p.id]) === "done").length;
  const started = modules.some((p) => status(p, saved[p.id]) !== "todo");
  return {
    completed,
    total: modules.length,
    state: completed > 0 && completed === modules.length ? "done" : started ? "started" : "todo",
  };
}

function StateIcon({ state }: { state: string }) {
  return <span className={`row-state ${state}`}>
    {state === "done" ? <CheckCircle2 size={18} /> : state === "started" ? <span className="in-progress-dot" /> : <Circle size={17} />}
  </span>;
}

function ProgressBar({ completed, total }: { completed: number; total: number }) {
  return <div className="paper-progress-bar" aria-hidden="true"><span style={{ width: `${total ? completed / total * 100 : 0}%` }} /></div>;
}

function PracticeFooter() {
  const { t } = useI18n();
  return <footer className="home-footer">
    <span>PaperLeet <i>·</i> Read. Implement. Understand.</span>
    <span>{t("代码和练习进度保存在当前浏览器。", "Code and practice progress are saved in this browser.")}</span>
  </footer>;
}

function TestingCard({ onGuide }: { onGuide: () => void }) {
  const { t } = useI18n();
  const labels = t("接口契约|参考对齐|数学性质|真实行为", "Contract|Reference|Properties|Behavior").split("|");
  const descriptions = [
    t("形状、类型和边界输入", "Shapes, types, and edge cases"),
    t("与参考结果逐项对齐", "Agreement with reference outputs"),
    t("把数学性质写成断言", "Mathematical properties as assertions"),
    t("验证训练或采样行为", "Training and sampling behavior"),
  ];
  return <section className="method-card">
    <div className="aside-title"><h3>{t("四层验证", "Four layers of testing")}</h3><Layers3 size={17} /></div>
    <p>{t("每个模块都有独立的接口和测试。", "Each module has its own contract and tests.")}</p>
    {layers.map((layer, i) => <div className="method-step" key={layer}><b>{layer}</b><div><strong>{labels[i]}</strong><span>{descriptions[i]}</span></div></div>)}
    <button className="text-button" onClick={onGuide}>{t("如何开始", "How to get started")} <ArrowRight size={14} /></button>
  </section>;
}

export function PaperLibrary({ papers, categories, problems, saved, onGuide }: Practice & { papers: Paper[]; categories: Category[]; onGuide: () => void }) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [diff, setDiff] = useState("all");
  const [state, setState] = useState("all");
  const [tag, setTag] = useState("all");
  const [category, setCategory] = useState("all");
  const practice = { problems, saved };
  const tags = [...new Set(papers.flatMap((paper) => paper.tags))];
  const completed = papers.filter((p) => paperProgress(p, practice).state === "done").length;
  const started = papers.filter((p) => paperProgress(p, practice).state === "started").length;
  const solvedModules = problems.filter((p) => status(p, saved[p.id]) === "done").length;
  const filtered = papers.filter((paper) => {
    const modules = paperModules(paper, problems);
    const searchable = [paper.number, paper.title, paper.short_title, paper.summary, paper.summary_en, ...paper.tags,
      ...modules.flatMap((p) => [p.title, p.translations?.en?.title, p.module.goal, p.module.goal_en])].join(" ").toLowerCase();
    return (category === "all" || paper.category === category) &&
      (diff === "all" || modules.some((p) => p.difficulty === diff)) &&
      (state === "all" || paperProgress(paper, practice).state === state) &&
      (tag === "all" || paper.tags.includes(tag)) && searchable.includes(query.trim().toLowerCase());
  });
  function clear() { setQuery(""); setDiff("all"); setState("all"); setTag("all"); setCategory("all"); }
  return <main className="home paper-library">
    <div className="home-intro">
      <div>
        <div className="eyebrow"><span />PAPER TO CODE</div>
        <h1>{t("读懂论文，", "Understand papers.")}<span>{t("从核心实现开始。", " Build the core ideas.")}</span></h1>
        <p>{t("选择一篇论文，实现其中精选的核心模块，用测试验证你的理解。", "Choose a paper, implement its selected core modules, and test your understanding.")}</p>
      </div>
      <a className="button subtle" href="#/submit"><Plus size={15} />{t("提交论文", "Submit paper")}</a>
    </div>
    <div className="paper-categories" aria-label={t("论文分类", "Paper categories")}>
      <button aria-pressed={category === "all"} onClick={() => { setCategory("all"); setTag("all"); }}>
        <span>{t("全部论文", "All papers")}</span><b>{papers.length}</b>
      </button>
      {categories.map((item) => <button key={item.id} aria-pressed={category === item.id} onClick={() => { setCategory(item.id); setTag("all"); }}>
        <span>{t(item.name, item.name_en)}</span><b>{papers.filter((paper) => paper.category === item.id).length}</b>
      </button>)}
    </div>
    <div className="home-columns">
      <div className="main-column">
        <section className="problem-list">
          <div className="list-title"><h2>{category === "all" ? t("全部论文", "All papers") : categories.filter((item) => item.id === category).map((item) => t(item.name, item.name_en))} <span>{filtered.length}</span></h2><span>Python · PyTorch</span></div>
          <div className="filter-bar">
            <div className="select-wrap"><select aria-label={t("主题筛选", "Filter by topic")} value={tag} onChange={(e) => setTag(e.target.value)}>
              <option value="all">{t("全部主题", "All topics")}</option>{tags.filter((name) => papers.some((paper) => (category === "all" || paper.category === category) && paper.tags.includes(name))).map((name) => <option key={name}>{name}</option>)}
            </select><ChevronDown size={12} /></div>
            <div className="select-wrap"><ListFilter size={14} /><select aria-label={t("模块难度筛选", "Filter module difficulty")} value={diff} onChange={(e) => setDiff(e.target.value)}>
              <option value="all">{t("全部难度", "All difficulties")}</option><option value="E">Easy</option><option value="M">Medium</option><option value="H">Hard</option>
            </select><ChevronDown size={12} /></div>
            <div className="select-wrap"><select aria-label={t("论文状态筛选", "Filter paper status")} value={state} onChange={(e) => setState(e.target.value)}>
              <option value="all">{t("全部状态", "All statuses")}</option><option value="todo">{t("未开始", "Not started")}</option><option value="started">{t("进行中", "In progress")}</option><option value="done">{t("核心模块已完成", "Core modules completed")}</option>
            </select><ChevronDown size={12} /></div>
          </div>
          <label className="search"><Search size={17} /><input aria-label={t("搜索论文或模块", "Search papers or modules")} placeholder={t("搜索论文编号、标题或核心模块…", "Search paper number, title, or core module…")} value={query} onChange={(e) => setQuery(e.target.value)} /><kbd>⌕</kbd></label>
          <div className="table-head paper-table-grid"><span>{t("状态", "Status")}</span><span>{t("论文", "Paper")}</span><span>{t("核心模块", "Core modules")}</span><span /></div>
          <div className="problem-rows">
            {filtered.map((paper) => {
              const progress = paperProgress(paper, practice);
              return <a className="problem-row paper-table-grid" key={paper.id} href={`#/papers/${paper.id}`}>
                <StateIcon state={progress.state} />
                <div className="problem-info paper-info"><strong><span>{paper.number}.</span>{paper.short_title}</strong><small>{paper.title}</small><div className="paper-row-tags">{paper.year && <span>{paper.year}</span>}{paper.tags.map((name) => <span key={name}>{name}</span>)}</div></div>
                <div className={`paper-row-progress ${progress.state}`}><span><b>{progress.completed}</b> / {progress.total}</span><ProgressBar {...progress} /></div>
                <ChevronRight size={16} />
              </a>;
            })}
          </div>
          {!filtered.length && <div className="no-results"><Search size={25} /><p>{t("没有符合条件的论文", "No matching papers")}</p><button className="text-button" onClick={clear}>{t("清除筛选", "Clear filters")}</button></div>}
          <div className="table-footer"><span>{t("显示", "Showing")} {filtered.length} / {papers.length}</span><span>{t("一篇论文，几个关键实现。", "A paper. Its essential implementations.")}</span></div>
        </section>
      </div>
      <aside className="home-sidebar">
        <section className="progress-card">
          <div className="aside-title"><h3>{t("你的进度", "Your progress")}</h3><span>{t("自动保存", "Autosaved")}</span></div>
          <div className="progress-circle" style={{ background: `conic-gradient(var(--accent) ${papers.length ? completed / papers.length * 360 : 0}deg, var(--line) 0deg)` }}><div><b>{completed}<span> / {papers.length}</span></b><small>{t("论文核心已完成", "PAPERS · CORE COMPLETE")}</small></div></div>
          <div className="progress-stats"><div><i className="green-dot" /><b>{completed}</b><span>{t("已完成", "Complete")}</span></div><div><i className="yellow-dot" /><b>{started}</b><span>{t("进行中", "Started")}</span></div><div><i /><b>{papers.length - completed - started}</b><span>{t("未开始", "To do")}</span></div></div>
          <div className="module-total"><FileCode2 size={14} /><span>{t("已通过模块", "Modules solved")}</span><b>{solvedModules} / {problems.length}</b></div>
        </section>
        <TestingCard onGuide={onGuide} />
        <section className="start-card"><span className="tiny-label">PAPER → CORE MODULES</span><h3>{t("贡献一篇论文", "Contribute a paper")}</h3><p>{t("提交论文与核心实现建议，经人工评审后进入题库。", "Submit a paper and a focused implementation for human review.")}</p><a href="#/submit">{t("提交论文", "Submit paper")} <ArrowRight size={15} /></a></section>
      </aside>
    </div>
    <PracticeFooter />
  </main>;
}

export function PaperDetail({ paper, problems, saved, onGuide }: Practice & { paper: Paper; onGuide: () => void }) {
  const { t, locale } = useI18n();
  const modules = paperModules(paper, problems);
  const progress = paperProgress(paper, { problems, saved });
  const next = modules.find((p) => !p.module.optional && status(p, saved[p.id]) !== "done") || modules[0];
  const authors = Array.isArray(paper.authors) ? paper.authors.join(", ") : paper.authors;
  return <main className="home paper-detail">
    <a className="back-link paper-library-link" href="#"><ArrowLeft size={15} />{t("全部论文", "All papers")}</a>
    <div className="paper-detail-header">
      <div><div className="eyebrow"><BookOpen size={14} />{t("论文", "PAPER")} {paper.number}</div><h1>{paper.short_title}</h1><p className="paper-full-title">{paper.title}</p><div className="paper-detail-meta">{paper.venue && <span>{paper.venue}</span>}{paper.year && <span>{paper.year}</span>}{paper.tags.map((tag) => <span className="paper-topic" key={tag}>{tag}</span>)}</div></div>
      <a className="button subtle" href={paper.url} target="_blank" rel="noreferrer"><BookOpen size={15} />{t("阅读论文", "Read paper")}<ExternalLink size={13} /></a>
    </div>
    <div className="home-columns">
      <div className="main-column">
        <section className="paper-overview"><div className="aside-title"><h2>{t("核心思想", "The core idea")}</h2><Layers3 size={18} /></div><p>{t(paper.summary, paper.summary_en)}</p><div className="paper-scope"><b>{t("本篇练习范围", "Selected practice scope")}</b><p>{t(paper.scope, paper.scope_en)}</p></div></section>
        <section className="paper-module-list">
          <div className="list-title"><h2>{t("实现核心模块", "Implement the core modules")}</h2><span>{progress.completed} / {progress.total} {t("已通过", "solved")}</span></div>
          <p className="module-list-intro">{t("每个模块独立实现与验证，完成全部必做模块即可完成本篇核心练习。", "Implement and test each module independently. Finish the required modules to complete this paper’s selected core practice.")}</p>
          {modules.map((problem) => {
            const title = locale === "en" ? problem.translations?.en?.title || problem.title : problem.title;
            const state = status(problem, saved[problem.id]);
            return <a href={`#/${problem.number}`} className="paper-module-card" key={problem.id}>
              <div className="module-card-top"><StateIcon state={state} /><h3><span>{problem.number}.</span> {title}</h3><ChevronRight size={17} /></div>
              <p>{t(problem.module.goal, problem.module.goal_en)}</p>
              <div className="module-card-tags"><span className={`difficulty ${problem.difficulty}`}>{({ E: "Easy", M: "Medium", H: "Hard" } as Record<string, string>)[problem.difficulty] || problem.difficulty}</span><span>{problem.estimated_minutes || 25} min</span><span>{problem.module.optional ? t("可选拓展", "Optional extension") : t("核心模块", "Core module")}</span>{state === "done" && <span className="accepted">{t("已通过", "Solved")}</span>}</div>
              <div className="module-functions"><span>{t("需要实现", "Implement")}</span>{problem.module.required_functions.map((name) => <code key={name}>{name}</code>)}</div>
              {!!problem.module.provided_functions.length && <div className="module-functions provided-functions"><span>{t("已提供", "Provided")}</span>{problem.module.provided_functions.map((name) => <code key={name}>{name}</code>)}</div>}
            </a>;
          })}
        </section>
      </div>
      <aside className="home-sidebar">
        <section className="paper-completion-card"><div className="aside-title"><h3>{t("论文进度", "Paper progress")}</h3><StateIcon state={progress.state} /></div><div className="paper-completion-count"><b>{progress.completed}</b><span>/ {progress.total}</span></div><p>{t("核心模块已通过", "core modules solved")}</p><ProgressBar {...progress} />{progress.state === "done" && <div className="core-complete"><CheckCircle2 size={16} />{t("本篇核心模块已完成", "Selected core modules complete")}</div>}{next && <a className="button" href={`#/${next.number}`}>{progress.state === "done" ? t("回顾模块", "Review modules") : progress.state === "started" ? t("继续实现", "Continue practice") : t("开始实现", "Start implementing")}<ArrowRight size={15} /></a>}<small>{t("进度按本篇精选的核心模块计算。", "Progress covers this paper’s selected core modules.")}</small></section>
        <section className="paper-source-card"><div className="aside-title"><h3>{t("论文信息", "Paper details")}</h3><BookOpen size={16} /></div><p>{authors}</p><a className="text-button" href={paper.url} target="_blank" rel="noreferrer">{paper.arxiv ? `arXiv:${paper.arxiv}` : t("原始论文", "Original paper")}<ExternalLink size={13} /></a></section>
        <TestingCard onGuide={onGuide} />
      </aside>
    </div>
    <PracticeFooter />
  </main>;
}
