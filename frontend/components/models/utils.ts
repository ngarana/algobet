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
  // Data range
  startDate: "",
  endDate: "",
  minMatches: 100,
  // Tournament and team filtering
  tournamentIds: [] as number[],
  teamIds: [] as number[],
  venueFilter: "both" as const,
  requireOdds: true,
  // Match quality filters
  minTotalGoals: null as number | null,
  maxTotalGoals: null as number | null,
  // Split ratios
  trainRatio: 0.7,
  valRatio: 0.15,
  testRatio: 0.15,
  // Training settings
  randomSeed: 42,
  earlyStoppingRounds: 50,
  tuningTrials: 50,
  // Calibration settings
  calibrateProbabilities: true,
  calibrationMethod: "isotonic" as const,
  // Outcome balancing
  outcomeBalance: true,
  // Feature groups (empty = all)
  featureGroups: [] as string[],
};
