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
  useGpuWorker: false,
  // Data range
  startDate: "",
  endDate: "",
  minMatches: 100,
  // Tournament and team filtering
  tournamentIds: [] as number[],
  teamIds: [] as number[],
  venueFilter: "both" as const,
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
  calibrationMethod: "sigmoid" as const,
  // Outcome balancing
  outcomeBalance: false,
  // Feature groups (empty = all)
  featureGroups: [] as string[],
  // Feature selection
  featureSelection: false,
  featureSelectionThreshold: 0.005,
  minSamplesPerFeature: 40 as number | null,
  // Ensemble training
  useEnsemble: false,
  ensembleTypes: ["xgboost", "lightgbm"] as string[],
  // Split strategy
  splitStrategy: "temporal" as const,
  gapDays: 0,
  // Expanding window params
  minTrainSize: 100,
  ewValSize: 50,
  ewTestSize: 50,
  stepSize: 50,
  // Season-aware params
  trainSeasons: 3,
  valSeasons: 1,
  testSeasons: 1,
  // Custom hyperparameters
  customHyperparameters: {} as Record<string, number>,
};
