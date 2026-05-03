"use client";

import { useState } from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { BasicSettings } from "./BasicSettings";
import { DataRangeSection } from "./DataRangeSection";
import { DataSelectionSection } from "./DataSelectionSection";
import { MatchQualitySection } from "./MatchQualitySection";
import { DataSplitSection } from "./DataSplitSection";
import { SplitStrategySection } from "./SplitStrategySection";
import { FeatureGroupsSection } from "./FeatureGroupsSection";
import { OutcomeBalancingSection } from "./OutcomeBalancingSection";
import { EnsembleSection } from "./EnsembleSection";
import { HyperparametersSection } from "./HyperparametersSection";
import { TrainingSettingsSection } from "./TrainingSettingsSection";
import { TrainingSummary } from "./TrainingSummary";
import type { TrainingConfig } from "./types";
import type { TrainingConfig as TrainingConfigType } from "./types";

interface GuidedTrainingWorkspaceProps {
  config: TrainingConfig;
  onConfigChange: <K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K]
  ) => void;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  isTraining: boolean;
  error: string | null;
}

export function GuidedTrainingWorkspace({
  config,
  onConfigChange,
  onSubmit,
  isTraining,
  error,
}: GuidedTrainingWorkspaceProps) {
  const [expandedSections, setExpandedSections] = useState<string[]>(["basics"]);

  return (
    <div className="space-y-4">
      <TrainingSummary config={config} isTraining={isTraining} />

      <form onSubmit={onSubmit} className="space-y-3">
        {error && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4" />
            <span>{error}</span>
          </div>
        )}

        <Accordion
          type="multiple"
          value={expandedSections}
          onValueChange={(values) => setExpandedSections(values)}
        >
          <AccordionItem value="basics" className="rounded-lg border">
            <AccordionTrigger className="px-4 py-3 hover:no-underline">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Basics</span>
                <BasicsSummary config={config} />
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-4 pb-4">
              <BasicSettings config={config} onConfigChange={onConfigChange} />
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="data-scope" className="rounded-lg border">
            <AccordionTrigger className="px-4 py-3 hover:no-underline">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Data Scope</span>
                <DataScopeSummary config={config} />
              </div>
            </AccordionTrigger>
            <AccordionContent className="space-y-6 px-4 pb-4">
              <DataRangeSection config={config} onConfigChange={onConfigChange} />
              <div className="border-t" />
              <DataSelectionSection config={config} onConfigChange={onConfigChange} />
              <div className="border-t" />
              <MatchQualitySection config={config} onConfigChange={onConfigChange} />
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="validation" className="rounded-lg border">
            <AccordionTrigger className="px-4 py-3 hover:no-underline">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Validation</span>
                <ValidationSummary config={config} />
              </div>
            </AccordionTrigger>
            <AccordionContent className="space-y-6 px-4 pb-4">
              <DataSplitSection config={config} onConfigChange={onConfigChange} />
              <div className="border-t" />
              <SplitStrategySection config={config} onConfigChange={onConfigChange} />
              <div className="border-t" />
              <OutcomeBalancingSection
                config={config}
                onConfigChange={onConfigChange}
              />
              {config.calibrateProbabilities && (
                <>
                  <div className="border-t" />
                  <div className="space-y-2">
                    <span className="text-sm font-semibold">Calibration Method</span>
                    <div className="flex gap-2">
                      {(["isotonic", "sigmoid"] as const).map((method) => (
                        <Badge
                          key={method}
                          variant={
                            config.calibrationMethod === method ? "default" : "outline"
                          }
                          className="cursor-pointer"
                          onClick={() => onConfigChange("calibrationMethod", method)}
                        >
                          {method === "isotonic" ? "Isotonic" : "Sigmoid"}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="expert" className="rounded-lg border">
            <AccordionTrigger className="px-4 py-3 hover:no-underline">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Expert Options</span>
                <ExpertSummary config={config} />
              </div>
            </AccordionTrigger>
            <AccordionContent className="space-y-6 px-4 pb-4">
              <FeatureGroupsSection config={config} onConfigChange={onConfigChange} />
              <div className="border-t" />
              <EnsembleSection config={config} onConfigChange={onConfigChange} />
              {!config.tune && !config.useEnsemble && (
                <>
                  <div className="border-t" />
                  <HyperparametersSection
                    config={config}
                    onConfigChange={onConfigChange}
                  />
                </>
              )}
              <div className="border-t" />
              <TrainingSettingsSection
                config={config}
                onConfigChange={onConfigChange}
              />
              {config.tune && (
                <div className="space-y-2">
                  <span className="text-sm font-semibold">Tuning Trials</span>
                  <span className="text-xs text-muted-foreground">
                    Will run {config.tuningTrials} Optuna trials before final training
                  </span>
                </div>
              )}
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        <div className="pt-2">
          <Button className="w-full" disabled={isTraining} type="submit">
            <Play className="mr-2 h-4 w-4" />
            {isTraining ? "Training..." : "Train Model"}
          </Button>
        </div>
      </form>
    </div>
  );
}

function BasicsSummary({ config }: { config: TrainingConfigType }) {
  const items: string[] = [];
  items.push(config.modelType);
  if (config.tune) items.push("Tuning");
  if (config.calibrateProbabilities) items.push("Calibrated");
  if (config.activate) items.push("Auto-activate");

  return (
    <Badge variant="secondary" className="text-xs">
      {items.join(" • ")}
    </Badge>
  );
}

function DataScopeSummary({ config }: { config: TrainingConfigType }) {
  const items: string[] = [];

  if (config.startDate || config.endDate) {
    items.push("Date range set");
  }

  if (config.minMatches > 100) {
    items.push(`Min ${config.minMatches} matches`);
  }

  if (config.tournamentIds.length > 0) {
    items.push(`${config.tournamentIds.length} tournament(s)`);
  }

  if (config.teamIds.length > 0) {
    items.push(`${config.teamIds.length} team(s)`);
  }

  if (config.venueFilter !== "both") {
    items.push(config.venueFilter === "home" ? "Home only" : "Away only");
  }

  if (config.requireOdds) {
    items.push("Odds required");
  }

  if (config.minTotalGoals !== null || config.maxTotalGoals !== null) {
    items.push("Goals filtered");
  }

  if (items.length === 0) {
    return <Badge variant="outline">All data</Badge>;
  }

  return (
    <Badge variant="secondary" className="text-xs">
      {items.length} filter(s)
    </Badge>
  );
}

function ValidationSummary({ config }: { config: TrainingConfigType }) {
  const totalRatio = config.trainRatio + config.valRatio + config.testRatio;
  const isValid = Math.abs(totalRatio - 1.0) <= 0.001;

  const items: string[] = [];

  if (!isValid) {
    return (
      <Badge variant="destructive" className="text-xs">
        Invalid split ratios
      </Badge>
    );
  }

  items.push(
    `${(config.trainRatio * 100).toFixed(0)}/${(config.valRatio * 100).toFixed(0)}/${(config.testRatio * 100).toFixed(0)}`
  );

  const strategyNames: Record<string, string> = {
    temporal: "Temporal",
    expanding_window: "Expanding",
    season_aware: "Season",
  };
  items.push(strategyNames[config.splitStrategy] || config.splitStrategy);

  if (config.outcomeBalance) {
    items.push("Balanced");
  }

  if (config.calibrateProbabilities) {
    items.push(config.calibrationMethod);
  }

  return (
    <Badge variant="secondary" className="text-xs">
      {items.join(" • ")}
    </Badge>
  );
}

function ExpertSummary({ config }: { config: TrainingConfigType }) {
  const items: string[] = [];

  if (config.featureGroups.length > 0) {
    items.push(`${config.featureGroups.length} features`);
  }

  if (config.useEnsemble) {
    items.push(`Ensemble (${config.ensembleTypes.join(", ")})`);
  }

  if (Object.keys(config.customHyperparameters).length > 0) {
    items.push("Custom params");
  }

  items.push(`Seed: ${config.randomSeed}`);

  if (config.earlyStoppingRounds > 0) {
    items.push(`Early stop: ${config.earlyStoppingRounds}`);
  }

  if (items.length <= 1) {
    return <Badge variant="outline">Defaults</Badge>;
  }

  return (
    <Badge variant="secondary" className="text-xs">
      {items.length - 1} setting(s)
    </Badge>
  );
}
