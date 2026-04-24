import type { ModelVersion } from "@/lib/types/api";
import type { GeneratePredictionsResult } from "@/lib/api/predictions";
import { Play, Check } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface PredictionControlsProps {
  activeModel: ModelVersion | null;
  models: ModelVersion[];
  selectedModel: ModelVersion | null;
  selectedModelId: string;
  daysAhead: number;
  generationResult: GeneratePredictionsResult | null;
  isGenerating: boolean;
  isActivating: boolean;
  onChangeDaysAhead: (value: number) => void;
  onChangeSelectedModelId: (value: string) => void;
  onGenerate: () => void;
  onActivate: () => void;
}

export default function PredictionControls({
  activeModel,
  models,
  selectedModel,
  selectedModelId,
  daysAhead,
  generationResult,
  isGenerating,
  isActivating,
  onChangeDaysAhead,
  onChangeSelectedModelId,
  onGenerate,
  onActivate,
}: PredictionControlsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Prediction Controls
        </CardTitle>
        <CardDescription>
          Choose the model you want to use, generate upcoming predictions, and
          optionally make it the active default.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_180px_auto_auto]">
          <div className="space-y-2">
            <Label htmlFor="prediction-model">Model</Label>
            <Select value={selectedModelId} onValueChange={onChangeSelectedModelId}>
              <SelectTrigger id="prediction-model">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {models.map((model) => (
                  <SelectItem key={model.id} value={String(model.id)}>
                    {model.version}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="days-ahead">Days Ahead</Label>
            <Select
              value={String(daysAhead)}
              onValueChange={(value) => onChangeDaysAhead(Number(value))}
            >
              <SelectTrigger id="days-ahead">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="3">3 days</SelectItem>
                <SelectItem value="7">7 days</SelectItem>
                <SelectItem value="14">14 days</SelectItem>
                <SelectItem value="30">30 days</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button
            className="self-end"
            disabled={!selectedModel || isGenerating}
            onClick={onGenerate}
          >
            {isGenerating ? "Generating..." : "Generate"}
          </Button>

          <Button
            className="self-end"
            disabled={!selectedModel || selectedModel.is_active || isActivating}
            onClick={onActivate}
            variant="outline"
          >
            {isActivating ? "Activating..." : "Set Active"}
          </Button>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
          {selectedModel ? (
            <>
              <Badge variant="secondary">{selectedModel.algorithm}</Badge>
              <span>Selected: {selectedModel.version}</span>
              {activeModel && activeModel.id === selectedModel.id && (
                <Badge className="bg-green-600">
                  <Check className="mr-1 h-3 w-3" />
                  Active
                </Badge>
              )}
            </>
          ) : (
            <span>No model selected.</span>
          )}
        </div>

        {generationResult && (
          <div className="rounded-md border bg-muted/30 p-4">
            <div className="text-sm font-medium">Latest generation run</div>
            <div className="mt-2 grid gap-3 md:grid-cols-4">
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground">
                  Model
                </div>
                <div className="font-mono text-sm">
                  {generationResult.model_version}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground">
                  Generated
                </div>
                <div className="text-lg font-semibold">
                  {generationResult.generated}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground">
                  Skipped
                </div>
                <div className="text-lg font-semibold">
                  {generationResult.existing_predictions_skipped}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground">
                  Processed
                </div>
                <div className="text-lg font-semibold">
                  {generationResult.matches_processed}
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
