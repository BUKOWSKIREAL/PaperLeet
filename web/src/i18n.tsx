import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

export type Locale = "zh" | "en";
type I18n = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (zh: string, en: string) => string;
};
const Context = createContext<I18n>({
  locale: "zh",
  setLocale: () => {},
  t: (zh) => zh,
});
export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>(() => {
    try {
      return localStorage.getItem("paperleet.locale") === "en" ? "en" : "zh";
    } catch {
      return "zh";
    }
  });
  useEffect(() => {
    document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
    try {
      localStorage.setItem("paperleet.locale", locale);
    } catch {
      /* Keep the session usable without storage. */
    }
  }, [locale]);
  return (
    <Context.Provider
      value={{ locale, setLocale, t: (zh, en) => (locale === "zh" ? zh : en) }}
    >
      {children}
    </Context.Provider>
  );
}
export const useI18n = () => useContext(Context);
