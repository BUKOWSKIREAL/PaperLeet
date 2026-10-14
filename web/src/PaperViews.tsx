import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft, ArrowRight, BookOpen, CheckCircle2, ChevronDown,
  ChevronRight, Circle, ExternalLink, Search, Library, Layers3, Network, SlidersHorizontal, Waves,
} from "lucide-react";
import { useI18n } from "./i18n";
import { status } from "./types";
import type { Category, Paper, Problem, Saved } from "./types";
import { paperLists, papersInList } from "./paperLists";

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
  const { t } = useI18n();
  const label = state === "done" ? t("已完成", "Solved") : state === "started" ? t("进行中", "In progress") : t("未开始", "Not started");
  return <span className={`row-state ${state}`} role="img" aria-label={label} title={label}>
    {state === "done" ? <CheckCircle2 size={18} /> : state === "started" ? <span className="in-progress-dot" /> : <Circle size={17} />}
  </span>;
}

function ProgressBar({ completed, total }: { completed: number; total: number }) {
  return <div className="paper-progress-bar" aria-hidden="true"><span style={{ width: `${total ? completed / total * 100 : 0}%` }} /></div>;
}

const categoryLabels: Record<string, [string, string]> = {
  "reinforcement-learning": ["强化学习", "RL"],
  "post-training": ["后训练与微调", "Post-training"],
  "generative-models": ["生成模型", "Generative models"],
  architecture: ["架构与推理", "Architecture"],
  optimization: ["优化算法", "Optimization"],
  representation: ["表征学习", "Representation"],
};

const listIcons = { "llm-basics": BookOpen, rl: Network, diffusion: Waves, "post-training": SlidersHorizontal, "visual-generation": Layers3 };

export function PaperLibrary({ papers, categories, problems, saved, listId }: Practice & { papers: Paper[]; categories: Category[]; onGuide: () => void; listId?: string }) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [diff, setDiff] = useState("all");
  const [state, setState] = useState("all");
  const [tag, setTag] = useState("all");
  const [category, setCategory] = useState("all");
  const practice = { problems, saved };
  const selectedList = paperLists.find((list) => list.id === listId);
  const listNavigation = useRef<HTMLElement>(null);
  useEffect(() => {
    const nav = listNavigation.current;
    if (nav && nav.scrollWidth > nav.clientWidth) {
      nav.querySelector('[aria-current="page"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }, [selectedList?.id]);
  const listedPapers = papersInList(papers, selectedList);
  const listedModuleIds = new Set(listedPapers.flatMap((paper) => paper.module_ids));
  const listedModules = problems.filter((problem) => listedModuleIds.has(problem.id));
  const visibleCategories = categories.filter((item) => listedPapers.some((paper) => paper.category === item.id));
  const tags = [...new Set(listedPapers.filter((paper) => category === "all" || paper.category === category).flatMap((paper) => paper.tags))];
  const solvedModules = listedModules.filter((p) => status(p, saved[p.id]) === "done").length;
  const filtered = listedPapers.filter((paper) => {
    const modules = paperModules(paper, problems);
    const searchable = [paper.number, paper.title, paper.short_title, paper.summary, paper.summary_en, ...paper.tags,
      ...modules.flatMap((p) => [p.title, p.translations?.en?.title, p.module.goal, p.module.goal_en])].join(" ").toLowerCase();
    return (category === "all" || paper.category === category) &&
      (diff === "all" || modules.some((p) => p.difficulty === diff)) &&
      (state === "all" || paperProgress(paper, practice).state === state) &&
      (tag === "all" || paper.tags.includes(tag)) && searchable.includes(query.trim().toLowerCase());
  });
  function clear() { setQuery(""); setDiff("all"); setState("all"); setTag("all"); setCategory("all"); }
  const hasFilters = !!query || diff !== "all" || state !== "all" || tag !== "all" || category !== "all";
  return <main className="home paper-library">
    <aside className="library-sidebar">
      <nav ref={listNavigation} aria-label={t("题单导航", "Paper lists")}>
        <a className="library-list-link library-all" href="#" onClick={clear} aria-current={!selectedList ? "page" : undefined}>
          <Library size={18} aria-hidden="true" /><span>{t("全部论文", "All papers")}</span><span className="list-count">{papers.length}</span>
        </a>
        <h2>{t("学习题单", "Study lists")}</h2>
        <div className="study-list-links">
          {paperLists.map((list) => {
            const Icon = listIcons[list.id as keyof typeof listIcons] || BookOpen;
            const count = papersInList(papers, list).length;
            const label = t(list.name, list.name_en);
            return <a key={list.id} className={`library-list-link list-${list.id}`} href={`#/lists/${list.id}`} onClick={clear} aria-current={selectedList?.id === list.id ? "page" : undefined} aria-label={`${label} · ${count} ${t("篇论文", "papers")}`}>
              <Icon size={18} aria-hidden="true" /><span>{label}</span><span className="list-count" aria-hidden="true">{count}</span>
            </a>;
          })}
        </div>
      </nav>
    </aside>
    <div className="library-main">
    <header className="library-heading">
      <h1>{selectedList ? t(selectedList.name, selectedList.name_en) : t("论文题库", "Papers")} <span>{listedPapers.length}</span></h1>
      <div className="library-progress"><CheckCircle2 size={16} /><span>{t("已通过模块", "Modules solved")}</span><b>{solvedModules} / {listedModules.length}</b></div>
    </header>
    <div className="paper-categories" aria-label={t("论文分类", "Paper categories")}>
      <button aria-pressed={category === "all"} onClick={() => { setCategory("all"); setTag("all"); }}>{t("全部", "All")}</button>
      {visibleCategories.map((item) => {
        const label = categoryLabels[item.id] || [item.name, item.name_en];
        return <button key={item.id} aria-pressed={category === item.id} onClick={() => { setCategory(item.id); setTag("all"); }}>{t(label[0], label[1])}</button>;
      })}
    </div>
    <section className="problem-list" aria-label={t("论文列表", "Paper list")}>
      <div className="library-toolbar">
        <label className="search"><Search size={17} aria-hidden="true" /><input type="search" aria-label={t("搜索论文或模块", "Search papers or modules")} placeholder={t("搜索论文或模块…", "Search papers or modules…")} value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Escape") setQuery(""); }} /></label>
        <div className="filter-bar">
          <div className="select-wrap"><select aria-label={t("主题筛选", "Filter by topic")} value={tag} onChange={(e) => setTag(e.target.value)}>
            <option value="all">{t("主题", "Topic")}</option>{tags.map((name) => <option key={name}>{name}</option>)}
          </select><ChevronDown size={12} aria-hidden="true" /></div>
          <div className="select-wrap"><select aria-label={t("模块难度筛选", "Filter module difficulty")} value={diff} onChange={(e) => setDiff(e.target.value)}>
            <option value="all">{t("难度", "Difficulty")}</option><option value="E">{t("简单", "Easy")}</option><option value="M">{t("中等", "Medium")}</option><option value="H">{t("困难", "Hard")}</option>
          </select><ChevronDown size={12} aria-hidden="true" /></div>
          <div className="select-wrap"><select aria-label={t("论文状态筛选", "Filter paper status")} value={state} onChange={(e) => setState(e.target.value)}>
            <option value="all">{t("状态", "Status")}</option><option value="todo">{t("未开始", "Not started")}</option><option value="started">{t("进行中", "In progress")}</option><option value="done">{t("已完成", "Solved")}</option>
          </select><ChevronDown size={12} aria-hidden="true" /></div>
        </div>
      </div>
      <div className="library-results" aria-live="polite"><span>{filtered.length} {t("篇论文", filtered.length === 1 ? "paper" : "papers")}</span>{hasFilters && <button className="text-button" onClick={clear}>{t("重置筛选", "Reset filters")}</button>}</div>
      <div className="table-head paper-table-grid"><span>{t("状态", "Status")}</span><span>{t("论文", "Paper")}</span><span className="paper-year">{t("年份", "Year")}</span><span>{t("已通过 / 模块", "Solved / Modules")}</span><span /></div>
      <div className="problem-rows">
        {filtered.map((paper) => {
          const progress = paperProgress(paper, practice);
          return <a className="problem-row paper-table-grid" key={paper.id} href={`#/papers/${paper.id}`}>
            <StateIcon state={progress.state} />
            <div className="problem-info paper-info"><strong><span>{paper.number}.</span>{paper.short_title}</strong><small title={paper.title}>{paper.title}</small></div>
            <span className="paper-year">{paper.year || "—"}</span>
            <div className={`paper-row-progress ${progress.state}`}><b>{progress.completed}</b><span> / {progress.total}</span></div>
            <ChevronRight size={16} aria-hidden="true" />
          </a>;
        })}
      </div>
      {!filtered.length && <div className="no-results"><Search size={25} /><p>{t("没有符合条件的论文", "No matching papers")}</p><button className="text-button" onClick={clear}>{t("清除筛选", "Clear filters")}</button></div>}
    </section>
    </div>
  </main>;
}

export function PaperDetail({ paper, problems, saved }: Practice & { paper: Paper; onGuide: () => void }) {
  const { t, locale } = useI18n();
  const modules = paperModules(paper, problems);
  const progress = paperProgress(paper, { problems, saved });
  const next = modules.find((p) => !p.module.optional && status(p, saved[p.id]) !== "done") || modules[0];
  const authors = Array.isArray(paper.authors) ? paper.authors.join(", ") : paper.authors;
  return <main className="home paper-detail">
    <a className="back-link paper-library-link" href="#"><ArrowLeft size={15} />{t("全部论文", "All papers")}</a>
    <div className="paper-detail-header">
      <div><h1><span>{paper.number}.</span> {paper.short_title}</h1><p className="paper-full-title">{paper.title}</p><div className="paper-detail-meta">{paper.venue && <span>{paper.venue}</span>}{paper.year && !paper.venue?.includes(String(paper.year)) && <span>{paper.year}</span>}{paper.tags.map((tag) => <span className="paper-topic" key={tag}>{tag}</span>)}</div></div>
      <a className="button subtle" href={paper.url} target="_blank" rel="noreferrer"><BookOpen size={15} />{t("阅读论文", "Read paper")}<ExternalLink size={13} /></a>
    </div>
    <div className="home-columns">
      <div className="main-column">
        <section className="paper-overview"><p>{t(paper.summary, paper.summary_en)}</p><details className="paper-scope-details"><summary>{t("练习范围", "Practice scope")}</summary><p>{t(paper.scope, paper.scope_en)}</p></details></section>
        <section className="paper-module-list">
          <div className="list-title"><h2>{t("核心模块", "Core modules")}</h2><span>{progress.completed} / {progress.total} {t("已通过", "solved")}</span></div>
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
        <section className="paper-completion-card"><div className="aside-title"><h3>{t("论文进度", "Paper progress")}</h3><StateIcon state={progress.state} /></div><div className="paper-completion-count"><b>{progress.completed}</b><span>/ {progress.total}</span></div><p>{t("核心模块已通过", "core modules solved")}</p><ProgressBar {...progress} />{progress.state === "done" && <div className="core-complete"><CheckCircle2 size={16} />{t("本篇核心模块已完成", "Selected core modules complete")}</div>}{next && <a className="button" href={`#/${next.number}`}>{progress.state === "done" ? t("回顾模块", "Review modules") : progress.state === "started" ? t("继续实现", "Continue practice") : t("开始实现", "Start implementing")}<ArrowRight size={15} /></a>}</section>
        <section className="paper-source-card"><div className="aside-title"><h3>{t("论文信息", "Paper details")}</h3><BookOpen size={16} /></div><p>{authors}</p><a className="text-button" href={paper.url} target="_blank" rel="noreferrer">{paper.arxiv ? `arXiv:${paper.arxiv}` : t("原始论文", "Original paper")}<ExternalLink size={13} /></a></section>
      </aside>
    </div>
  </main>;
}
