import { useState } from "react";
import { ArrowLeft, ArrowRight, BookOpen, CheckCircle2, ChevronDown, ChevronRight, Circle, Search, Shuffle } from "lucide-react";
import CollectionArtwork from "./CollectionArtwork";
import { useI18n } from "./i18n";
import { paperCollections, papersInCollection } from "./paperCollections";
import type { PaperCollection } from "./paperCollections";
import { paperModules } from "./PaperViews";
import { eligibleUnattemptedProblems, pickRandomUnattemptedProblem } from "./randomProblem";
import { status } from "./types";
import type { Paper, Problem, Saved } from "./types";

export type CollectionFilters = { query: string; difficulty: string; state: string };

/** Filter individual modules while preserving both curated paper and module order. */
export function collectionModuleGroups(
  papers: Paper[],
  problems: Problem[],
  collection: PaperCollection,
  filters: CollectionFilters,
  saved: Record<string, Saved>,
  progressReady: boolean,
) {
  const query = filters.query.trim().toLowerCase();
  return papersInCollection(papers, collection).flatMap((paper) => {
    const paperMatches = [paper.short_title, paper.title, ...paper.tags].join(" ").toLowerCase().includes(query);
    const modules = paperModules(paper, problems).filter((problem) => {
      const matchesQuery = paperMatches || [problem.number, problem.id, problem.title, problem.translations?.en?.title, problem.module.goal, problem.module.goal_en]
        .join(" ").toLowerCase().includes(query);
      return matchesQuery &&
        (filters.difficulty === "all" || problem.difficulty === filters.difficulty) &&
        (!progressReady || filters.state === "all" || status(problem, saved[problem.id]) === filters.state);
    });
    return modules.length ? [{ paper, modules }] : [];
  });
}

export default function CollectionDetail({ collectionId, papers, problems, saved, progressReady }: {
  collectionId: string;
  papers: Paper[];
  problems: Problem[];
  saved: Record<string, Saved>;
  progressReady: boolean;
}) {
  const { t, locale } = useI18n();
  const [query, setQuery] = useState("");
  const [difficulty, setDifficulty] = useState("all");
  const [state, setState] = useState("all");
  const collection = paperCollections.find((item) => item.id === collectionId);
  const back = <a className="back-link" href="#"><ArrowLeft size={16} />{t("全部论文", "All papers")}</a>;

  if (!collection) return <main className="home collection-detail collection-not-found">
    {back}<BookOpen size={36} aria-hidden="true" />
    <h1>{t("未找到这个合集", "Collection not found")}</h1>
    <p>{t("这个合集不存在，请返回题库选择其他合集。", "This collection does not exist. Choose another collection from the paper library.")}</p>
    <a className="button" href="#">{t("返回题库", "Back to papers")}<ArrowRight size={16} /></a>
  </main>;

  const members = papersInCollection(papers, collection);
  const modules = members.flatMap((paper) => paperModules(paper, problems));
  const solved = progressReady ? modules.filter((problem) => status(problem, saved[problem.id]) === "done").length : undefined;
  const started = progressReady && modules.some((problem) => status(problem, saved[problem.id]) !== "todo");
  const next = progressReady ? modules.find((problem) => status(problem, saved[problem.id]) !== "done") || modules[0] : undefined;
  const groups = collectionModuleGroups(papers, problems, collection, { query, difficulty, state }, saved, progressReady);
  const filteredModules = groups.flatMap((group) => group.modules);
  const unattempted = progressReady ? eligibleUnattemptedProblems(filteredModules, saved) : [];
  const allSolved = solved !== undefined && modules.length > 0 && solved === modules.length;
  const nextLabel = allSolved ? t("复习题目", "Review problems") : started ? t("继续练习", "Continue") : t("开始练习", "Start practicing");
  const progressLabel = t("已通过题目", "Problems solved");
  function clear() { setQuery(""); setDifficulty("all"); setState("all"); }
  function randomProblem() {
    if (!progressReady) return;
    const problem = pickRandomUnattemptedProblem(filteredModules, saved);
    if (problem) location.hash = `/${problem.number}`;
  }
  const hasFilters = !!query || difficulty !== "all" || state !== "all";
  const statusLabel = (value: string) => value === "done" ? t("已通过", "Solved") : value === "started" ? t("进行中", "In progress") : t("未开始", "Not started");
  return <main className="home collection-detail">
    {back}
    <header className={`collection-cover collection-${collection.id}`}>
      <div className="collection-cover-copy">
        <span className="collection-cover-eyebrow">{t("精选合集", "Featured collection")}</span>
        <h1>{t(collection.name, collection.name_en)}</h1>
        <p>{t(collection.description, collection.description_en)}</p>
        <div className="collection-cover-counts"><BookOpen size={16} aria-hidden="true" /><span>{members.length} {t("篇论文", "papers")}</span><span aria-hidden="true">·</span><span>{modules.length} {t("道题", "problems")}</span></div>
      </div>
      <CollectionArtwork kind={collection.id} />
    </header>

    <section className="collection-practice-summary" aria-label={t("合集进度", "Collection progress")}>
      <div className="collection-progress-copy">
        <span>{progressLabel}</span>
        <strong>{solved === undefined ? "—" : solved}<small> / {modules.length}</small></strong>
      </div>
      <div className="collection-progress-track">
        {solved === undefined ? <p className="collection-progress-pending">{t("学习进度尚未就绪", "Practice progress is not ready")}</p> : <>
          <div className="collection-progress-caption"><span>{allSolved ? t("全部通过", "Collection complete") : t("你的进度", "Your progress")}</span><span>{modules.length ? Math.round(solved / modules.length * 100) : 0}%</span></div>
          <div className="paper-progress-bar" role="progressbar" aria-label={progressLabel} aria-valuemin={0} aria-valuemax={modules.length || 1} aria-valuenow={solved}><span style={{ width: `${modules.length ? solved / modules.length * 100 : 0}%` }} /></div>
        </>}
      </div>
      {next ? <a className="button collection-start" href={`#/${next.number}`}>{allSolved ? <CheckCircle2 size={17} /> : <ArrowRight size={17} />}{nextLabel}</a> : <button className="button collection-start" disabled>{nextLabel}<ArrowRight size={17} /></button>}
    </section>

    <section className="collection-problems" aria-label={t("合集题目", "Collection problems")}>
      <div className="library-toolbar">
        <label className="search"><Search size={17} aria-hidden="true" /><input type="search" aria-label={t("搜索合集中的论文或题目", "Search papers or problems in collection")} placeholder={t("搜索论文或题目…", "Search papers or problems…")} value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Escape") setQuery(""); }} /></label>
        <div className="filter-bar">
          <div className="select-wrap"><select aria-label={t("题目难度筛选", "Filter problem difficulty")} value={difficulty} onChange={(event) => setDifficulty(event.target.value)}>
            <option value="all">{t("全部难度", "All difficulties")}</option><option value="E">{t("简单", "Easy")}</option><option value="M">{t("中等", "Medium")}</option><option value="H">{t("困难", "Hard")}</option>
          </select><ChevronDown size={12} aria-hidden="true" /></div>
          <div className="select-wrap"><select aria-label={t("题目状态筛选", "Filter problem status")} value={progressReady ? state : "all"} disabled={!progressReady} onChange={(event) => setState(event.target.value)}>
            <option value="all">{t("全部状态", "All statuses")}</option><option value="todo">{t("未开始", "Not started")}</option><option value="started">{t("进行中", "In progress")}</option><option value="done">{t("已通过", "Solved")}</option>
          </select><ChevronDown size={12} aria-hidden="true" /></div>
        </div>
        <button className="random-problem" disabled={!progressReady || !unattempted.length} onClick={randomProblem} title={!progressReady ? t("学习进度尚未就绪", "Practice progress is not ready") : unattempted.length ? t(`从筛选后的 ${unattempted.length} 道未开始的题目中随机选择`, `Pick from ${unattempted.length} unattempted problems matching your filters`) : t("当前范围内没有未开始的题目", "No unattempted problems in this view")}><Shuffle size={17} />{t("随机一题", "Random problem")}</button>
      </div>
      <div className="library-results" aria-live="polite"><span>{filteredModules.length} {t("道题", filteredModules.length === 1 ? "problem" : "problems")} · {groups.length} {t("篇论文", groups.length === 1 ? "paper" : "papers")}</span>{hasFilters && <button className="text-button" onClick={clear}>{t("重置筛选", "Reset filters")}</button>}</div>
      <div className="collection-paper-groups">
        {groups.map(({ paper, modules: visibleModules }) => <section className="collection-paper-group" key={paper.id} aria-labelledby={`collection-paper-${paper.id}`}>
          <header className="collection-paper-heading"><div><h2 id={`collection-paper-${paper.id}`}><a href={`#/papers/${paper.id}`}>{paper.short_title}<ChevronRight size={16} aria-hidden="true" /></a></h2><p>{paper.title}</p></div><span>{visibleModules.length} {t("道题", visibleModules.length === 1 ? "problem" : "problems")}</span></header>
          <ul className="collection-module-list">
            {visibleModules.map((problem) => {
              const currentState = progressReady ? status(problem, saved[problem.id]) : "pending";
              const label = currentState === "pending" ? t("学习进度尚未就绪", "Practice progress is not ready") : statusLabel(currentState);
              return <li key={problem.id}><a className="collection-module-row" href={`#/${problem.number}`}>
                <span className={`row-state ${currentState}`} role="img" aria-label={label} title={label}>{currentState === "pending" ? <span aria-hidden="true">—</span> : currentState === "done" ? <CheckCircle2 size={18} /> : currentState === "started" ? <span className="in-progress-dot" /> : <Circle size={17} />}</span>
                <span className="collection-module-title"><span className="collection-module-number">{problem.number}.</span>{locale === "en" ? problem.translations?.en?.title || problem.title : problem.title}{problem.module.optional && <small>{t("拓展", "Optional")}</small>}</span>
                <span className={`difficulty ${problem.difficulty}`}>{problem.difficulty === "E" ? t("简单", "Easy") : problem.difficulty === "M" ? t("中等", "Medium") : problem.difficulty === "H" ? t("困难", "Hard") : problem.difficulty}</span>
                <ChevronRight size={16} className="collection-module-arrow" aria-hidden="true" />
              </a></li>;
            })}
          </ul>
        </section>)}
      </div>
      {!filteredModules.length && <div className="no-results"><Search size={27} aria-hidden="true" /><p>{t("没有符合条件的题目", "No matching problems")}</p>{hasFilters && <button className="text-button" onClick={clear}>{t("清除筛选", "Clear filters")}</button>}</div>}
    </section>
  </main>;
}
