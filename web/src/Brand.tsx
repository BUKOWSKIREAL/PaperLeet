import { useI18n } from "./i18n";

/** A soft, abstract black cat with a tiny smile, legible at icon sizes. */
export function BrandMark() {
  return <svg className="brand-symbol" viewBox="0 0 64 64" aria-hidden="true" focusable="false">
    <circle cx="32" cy="32" r="31" fill="#f3efe6" />
    <path d="M12 27 11.5 16Q11 11 15.5 13.5L24 19Q32 17 40 19L48.5 13.5Q53 11 52.5 16L52 27Q57 33 55 41Q52 54 32 55Q12 54 9 41Q7 33 12 27Z" fill="#161619" />
    <ellipse cx="23" cy="35" rx="3.5" ry="4.5" fill="#f3efe6" />
    <ellipse cx="41" cy="35" rx="3.5" ry="4.5" fill="#f3efe6" />
    <path d="M28 44q2 3 4 0 2 3 4 0" fill="none" stroke="#f3efe6" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
  </svg>;
}

export default function Brand() {
  const { t } = useI18n();
  return <a href="#" className="brand" aria-label={t("PaperLeet 首页", "PaperLeet home")}>
    <BrandMark />
    <span className="brand-wordmark">Paper<span>Leet</span></span>
  </a>;
}
