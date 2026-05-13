"use client";

import { useState, type FormEvent } from "react";
import { FlaskConical, Shuffle, Layers, Target } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useActiveModel, useModels } from "@/lib/queries/use-models";
import type { AblationRequest } from "@/lib/types/ablation";

const FEATURE_GROUPS = [
  "team_form",
  "head_to_head",
  "temporal",
  "standings",
  "enriched_stats",
];

interface AblationFormProps {
  onSubmit: (request: AblationRequest) => void;
  isLoading: boolean;
}

export function AblationForm({ onSubmit, isLoading }: AblationFormProps) {
  const { data: activeModel } = useActiveModel();
  useModels();

  const [method, setMethod] = useState<"permutation" | "ablation">("permutation");
  const [modelVersion, setModelVersion] = useState("");
  const [nRepeats, setNRepeats] = useState(10);
  const [groupBy, setGroupBy] = useState<"family" | "generator">("generator");
  const [selectedFamilies, setSelectedFamilies] = useState<string[]>([]);
  const [minMatches, setMinMatches] = useState(100);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const toggleFamily = (family: string) => {
    setSelectedFamilies((prev) =>
      prev.includes(family) ? prev.filter((f) => f !== family) : [...prev, family]
    );
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    onSubmit({
      method,
      model_version: modelVersion || undefined,
      n_repeats: method === "permutation" ? nRepeats : 10,
      random_state: 42,
      group_by: groupBy,
      feature_families: selectedFamilies.length > 0 ? selectedFamilies : undefined,
      min_matches: minMatches,
      start_date: startDate || undefined,
      end_date: endDate || undefined,
      train_ratio: 0.7,
      val_ratio: 0.15,
      test_ratio: 0.15,
      gap_days: 0,
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FlaskConical className="h-5 w-5" />
          Feature Importance Analysis
        </CardTitle>
        <CardDescription>
          Measure which feature families contribute most to predictions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
            <Target className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">
              Active Model:{" "}
              {activeModel ? (
                <span className="font-medium">
                  {activeModel.name} ({activeModel.algorithm})
                </span>
              ) : (
                <span className="text-destructive">No active model</span>
              )}
            </span>
          </div>

          <div className="space-y-2">
            <Label>Method</Label>
            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant={method === "permutation" ? "default" : "outline"}
                className="flex flex-col items-center gap-1 py-3"
                onClick={() => setMethod("permutation")}
              >
                <Shuffle className="h-4 w-4" />
                <span className="text-xs">Permutation</span>
                <span className="text-[10px] text-muted-foreground">
                  Fast, no retraining
                </span>
              </Button>
              <Button
                type="button"
                variant={method === "ablation" ? "default" : "outline"}
                className="flex flex-col items-center gap-1 py-3"
                onClick={() => setMethod("ablation")}
              >
                <Layers className="h-4 w-4" />
                <span className="text-xs">Ablation</span>
                <span className="text-[10px] text-muted-foreground">
                  Slow, retrains model
                </span>
              </Button>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="model-version">
              Model Version{" "}
              <span className="text-muted-foreground">(empty = active)</span>
            </Label>
            <Input
              id="model-version"
              placeholder="e.g. xgboost_20260510_123233"
              value={modelVersion}
              onChange={(e) => setModelVersion(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="group-by">Group Features By</Label>
            <Select
              value={groupBy}
              onValueChange={(v) => setGroupBy(v as "family" | "generator")}
            >
              <SelectTrigger id="group-by">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="generator">
                  Generator (team_form, h2h, etc.)
                </SelectItem>
                <SelectItem value="family">
                  Sub-family (form, away, draw, etc.)
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {method === "permutation" && (
            <div className="space-y-2">
              <Label htmlFor="n-repeats">Permutation Repeats</Label>
              <Input
                id="n-repeats"
                type="number"
                min={1}
                max={100}
                value={nRepeats}
                onChange={(e) => setNRepeats(parseInt(e.target.value) || 10)}
              />
            </div>
          )}

          <div className="space-y-2">
            <Label>Feature Families</Label>
            <p className="text-xs text-muted-foreground">
              Leave empty to evaluate all families
            </p>
            <div className="flex flex-wrap gap-2">
              {FEATURE_GROUPS.map((group) => (
                <Button
                  key={group}
                  type="button"
                  variant={selectedFamilies.includes(group) ? "default" : "outline"}
                  size="sm"
                  onClick={() => toggleFamily(group)}
                >
                  {group}
                </Button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="start-date">Start Date</Label>
              <Input
                id="start-date"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="end-date">End Date</Label>
              <Input
                id="end-date"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="min-matches">Minimum Matches</Label>
            <Input
              id="min-matches"
              type="number"
              min={10}
              max={10000}
              value={minMatches}
              onChange={(e) => setMinMatches(parseInt(e.target.value) || 100)}
            />
          </div>

          <Button type="submit" className="w-full" disabled={isLoading || !activeModel}>
            {isLoading ? (
              <>
                <span className="mr-2 animate-spin">...</span>
                {method === "permutation"
                  ? "Running Permutation..."
                  : "Running Ablation..."}
              </>
            ) : (
              <>
                {method === "permutation" ? (
                  <Shuffle className="mr-2 h-4 w-4" />
                ) : (
                  <Layers className="mr-2 h-4 w-4" />
                )}
                {method === "permutation"
                  ? "Run Permutation Analysis"
                  : "Run Ablation Study"}
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
