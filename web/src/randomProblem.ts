import type { Problem, Saved } from "./types";

/** A cached starter alone does not count as an attempt. */
export function eligibleUnattemptedProblems(
  problems: Problem[],
  saved: Record<string, Saved>,
): Problem[] {
  const seen = new Set<string>();
  return problems.filter((problem) => {
    if (seen.has(problem.id)) return false;
    seen.add(problem.id);
    const progress = saved[problem.id];
    return (
      !progress?.report &&
      !progress?.acceptedReport &&
      (progress?.code === undefined || progress.code === problem.starter)
    );
  });
}

/** The injected random source follows Math.random's [0, 1) contract. */
export function pickRandomUnattemptedProblem(
  problems: Problem[],
  saved: Record<string, Saved>,
  random: () => number = Math.random,
): Problem | undefined {
  const eligible = eligibleUnattemptedProblems(problems, saved);
  if (!eligible.length) return undefined;
  return eligible[Math.floor(random() * eligible.length)];
}
