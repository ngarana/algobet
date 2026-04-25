export function formatMetricValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (Array.isArray(value)) {
    return value.join(", ");
  }

  return String(value);
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(0)}s`;
  }
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs.toFixed(0)}s`;
}

export const defaultConfig = {
  modelType: "xgboost" as const,
  description: "",
  tune: false,
  activate: true,
  startDate: "",
  endDate: "",
  minMatches: 100,
  trainRatio: 0.7,
  valRatio: 0.15,
  testRatio: 0.15,
  randomSeed: 42,
  earlyStoppingRounds: 50,
  tuningTrials: 50,
  calibrateProbabilities: true,
  calibrationMethod: "isotonic" as const,
};
