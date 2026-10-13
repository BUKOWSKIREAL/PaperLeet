export function dpoObjective(margin: number, beta: number, smoothing: number) {
  const z = beta * margin;
  const softplus = (x: number) =>
    Math.max(0, x) + Math.log1p(Math.exp(-Math.abs(x)));
  const probability = 1 / (1 + Math.exp(-z));
  return {
    loss: (1 - smoothing) * softplus(-z) + smoothing * softplus(z),
    gradient: beta * (probability - (1 - smoothing)),
    probability,
  };
}

export type Aggregation = "GRPO" | "DAPO" | "Dr. GRPO";
export function tokenGradient(
  mode: Aggregation,
  advantage: number,
  length: number,
  lengths: number[],
  maxLength = 24,
) {
  const denominator =
    mode === "GRPO"
      ? lengths.length * length
      : mode === "DAPO"
        ? lengths.reduce((sum, n) => sum + n, 0)
        : lengths.length * maxLength;
  return -advantage / denominator;
}

export function difficultyAdvantage(
  p: number,
  sampleStd = true,
  groupSize = 8,
) {
  if (p <= 0 || p >= 1) return { grpo: 0, dr: 0 };
  return {
    grpo:
      Math.sqrt((1 - p) / p) *
      (sampleStd ? Math.sqrt((groupSize - 1) / groupSize) : 1),
    dr: 1 - p,
  };
}

export function scheduleData(kind: string, count = 1000, offset = 0.008) {
  const f = (t: number) =>
    Math.cos((((t + offset) / (1 + offset)) * Math.PI) / 2) ** 2;
  const betas = Array.from({ length: count }, (_, i) =>
    kind === "cosine"
      ? Math.min(0.999, 1 - f((i + 1) / count) / f(i / count))
      : 0.0001 + ((0.02 - 0.0001) * i) / (count - 1),
  );
  let cumulative = 1;
  const bars = betas.map((beta) => (cumulative *= 1 - beta));
  return { betas, bars };
}
