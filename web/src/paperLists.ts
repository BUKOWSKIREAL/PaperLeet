import type { Paper } from "./types";

export type PaperList = {
  id: string;
  name: string;
  name_en: string;
  paperIds: string[];
};

// Curated learning lists can overlap; each paper retains its catalog category.
export const paperLists: PaperList[] = [
  {
    id: "llm-basics",
    name: "LLM 基础",
    name_en: "LLM Basics",
    paperIds: ["transformer", "adamw", "flash-attention", "lora"],
  },
  {
    id: "rl",
    name: "强化学习",
    name_en: "RL",
    paperIds: [
      "gae", "ppo", "deepseekmath", "dr-grpo", "gspo", "dapo",
      "glm-5", "sdpo", "pkpo", "diffusion-nft",
    ],
  },
  {
    id: "diffusion",
    name: "扩散与 Flow",
    name_en: "Diffusion & Flow",
    paperIds: [
      "ddpm", "ddim", "progressive-distillation", "flow-matching",
      "consistency", "dmd", "dmd2", "meanflow", "jit", "repa",
      "block-diffusion", "svdquant", "daps", "diffusion-nft", "fd-loss",
    ],
  },
  {
    id: "post-training",
    name: "后训练",
    name_en: "Post-training",
    paperIds: ["lora", "qlora", "dpo", "gkd", "opsd", "sdpo", "distillm-2"],
  },
  {
    id: "visual-generation",
    name: "视觉生成",
    name_en: "Visual Generation",
    paperIds: ["clip", "rotation-trick", "var", "repa", "drifting", "fd-loss"],
  },
];

export function papersInList(papers: Paper[], list?: PaperList): Paper[] {
  if (!list) return papers;
  const byId = new Map(papers.map((paper) => [paper.id, paper]));
  return list.paperIds.flatMap((id) => {
    const paper = byId.get(id);
    return paper ? [paper] : [];
  });
}
