import type { TrainModelResult } from "@/lib/types/ml-operations";
import type { ModelVersion } from "@/lib/types/api";

export interface TrainingConfig {
  modelType: "xgboost" | "lightgbm" | "random_forest";
  description: string;
  tune: boolean;
  activate: boolean;
  // Data range
  startDate: string;
  endDate: string;
  minMatches: number;
  // Tournament and team filtering
  tournamentIds: number[];
  teamIds: number[];
  venueFilter: "home" | "away" | "both";
  requireOdds: boolean;
  // Match quality filters
  minTotalGoals: number | null;
  maxTotalGoals: number | null;
  // Split ratios
  trainRatio: number;
  valRatio: number;
  testRatio: number;
  // Training settings
  randomSeed: number;
  earlyStoppingRounds: number;
  tuningTrials: number;
  // Calibration settings
  calibrateProbabilities: boolean;
  calibrationMethod: "isotonic" | "sigmoid";
  // Outcome balancing
  outcomeBalance: boolean;
  // Feature groups
  featureGroups: string[];
}

export interface TrainingResultDisplayProps {
  result: TrainModelResult;
}

export interface ModelRowProps {
  model: ModelVersion;
  isActive: boolean;
  isExpanded: boolean;
  onActivate: (id: number) => void;
  onDelete: (id: number) => void;
  onToggleMetrics: (model: ModelVersion | null) => void;
}

export interface ModelMetricsPanelProps {
  model: ModelVersion;
  onClose: () => void;
}

export interface DataRangeSectionProps {
  config: TrainingConfig;
  onConfigChange: <K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K]
  ) => void;
}

export interface DataSplitSectionProps {
  config: TrainingConfig;
  onConfigChange: <K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K]
  ) => void;
}

export interface TrainingSettingsSectionProps {
  config: TrainingConfig;
  onConfigChange: <K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K]
  ) => void;
}

export interface BasicSettingsProps {
  config: TrainingConfig;
  onConfigChange: <K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K]
  ) => void;
}

export interface AdvancedSettingsProps {
  config: TrainingConfig;
  onConfigChange: <K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K]
  ) => void;
}
