import { createHighlighterCore } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import python from "@shikijs/langs/python";
import bash from "@shikijs/langs/bash";
import json from "@shikijs/langs/json";
import yaml from "@shikijs/langs/yaml";
import latex from "@shikijs/langs/latex";
import theme from "@shikijs/themes/github-dark";

const highlighter = createHighlighterCore({
  themes: [theme],
  langs: [python, bash, json, yaml, latex],
  engine: createJavaScriptRegexEngine(),
});

export async function highlight(code: string, language: string) {
  const engine = await highlighter;
  const lang = language === "tex" ? "latex" : language;
  return engine.codeToHtml(code, {
    lang: ["python", "bash", "json", "yaml", "latex"].includes(lang)
      ? lang
      : "text",
    theme: "github-dark",
  });
}
