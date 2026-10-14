import { useI18n } from "./i18n";

/** A folded research paper containing a three-node computational graph. */
export function BrandMark() {
  return <svg className="brand-symbol" viewBox="0 0 48 48" aria-hidden="true" focusable="false">
    <path className="brand-sheet" d="M28 5H13a5 5 0 0 0-5 5v28a5 5 0 0 0 5 5h22a5 5 0 0 0 5-5V17L28 5Z" />
    <path className="brand-fold" d="M28 5v8a4 4 0 0 0 4 4h8" />
    <path className="brand-graph" d="m16 33 8-10 8 10" />
    <circle className="brand-node" cx="16" cy="33" r="3" />
    <circle className="brand-node" cx="32" cy="33" r="3" />
    <circle className="brand-core" cx="24" cy="23" r="3.5" />
  </svg>;
}

export default function Brand() {
  const { t } = useI18n();
  return <a href="#" className="brand" aria-label={t("LeetPaper 首页", "LeetPaper home")}>
    <BrandMark />
    <span className="brand-wordmark">Leet<span>Paper</span></span>
  </a>;
}
