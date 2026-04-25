import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress, ProgressValue } from "@/components/ui/progress";
import {
  Brain,
  TrendingUp,
  Calendar,
  BarChart3,
  ArrowRight,
  CheckCircle2,
} from "lucide-react";
import type { ModelVersion } from "@/lib/types/api";

interface ModelPerformanceCardProps {
  model: ModelVersion;
  totalPredictions?: number;
  winRate?: number;
  roi?: number;
  className?: string;
}

export default function ModelPerformanceCard({
  model,
  totalPredictions = 0,
  winRate,
  roi,
  className,
}: ModelPerformanceCardProps) {
  const accuracy = model.accuracy ?? 0;
  const isActive = model.is_active;

  const getAccuracyColor = (acc: number) => {
    if (acc >= 0.7) return "text-green-600";
    if (acc >= 0.5) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <Card className={className}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">
          <span className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Model Performance
          </span>
        </CardTitle>
        {isActive && (
          <Badge variant="default" className="bg-green-600">
            <CheckCircle2 className="mr-1 h-3 w-3" />
            Active
          </Badge>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Model</span>
            <span className="text-sm font-medium">{model.version}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Algorithm</span>
            <Badge variant="outline">{model.algorithm}</Badge>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1 text-sm text-muted-foreground">
              <BarChart3 className="h-3 w-3" />
              Accuracy
            </span>
            <span className={`text-sm font-bold ${getAccuracyColor(accuracy)}`}>
              {(accuracy * 100).toFixed(1)}%
            </span>
          </div>
          <Progress className="h-2">
            <ProgressValue value={accuracy * 100} />
          </Progress>
        </div>

        {winRate !== undefined && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1 text-sm text-muted-foreground">
                <TrendingUp className="h-3 w-3" />
                Win Rate
              </span>
              <span className="text-sm font-medium">{(winRate * 100).toFixed(1)}%</span>
            </div>
            <Progress className="h-2">
              <ProgressValue value={winRate * 100} />
            </Progress>
          </div>
        )}

        {roi !== undefined && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">ROI</span>
            <span
              className={`text-sm font-bold ${roi >= 0 ? "text-green-600" : "text-red-600"}`}
            >
              {roi >= 0 ? "+" : ""}
              {roi.toFixed(2)}%
            </span>
          </div>
        )}

        <div className="flex items-center justify-between">
          <span className="flex items-center gap-1 text-sm text-muted-foreground">
            <Calendar className="h-3 w-3" />
            Created
          </span>
          <span className="text-sm">
            {new Date(model.created_at).toLocaleDateString()}
          </span>
        </div>

        {totalPredictions > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Predictions</span>
            <span className="text-sm font-medium">
              {totalPredictions.toLocaleString()}
            </span>
          </div>
        )}

        {model.metrics && (
          <div className="space-y-1 border-t pt-2">
            <span className="text-xs text-muted-foreground">Additional Metrics</span>
            <div className="flex flex-wrap gap-1">
              {Object.entries(model.metrics)
                .slice(0, 4)
                .map(([key, value]) => (
                  <Badge key={key} variant="secondary" className="text-xs">
                    {key}:{" "}
                    {typeof value === "number" ? value.toFixed(3) : String(value)}
                  </Badge>
                ))}
            </div>
          </div>
        )}

        <Button variant="outline" size="sm" className="w-full" asChild>
          <a href={`/models?id=${model.id}`}>
            View Full Details
            <ArrowRight className="ml-2 h-3 w-3" />
          </a>
        </Button>
      </CardContent>
    </Card>
  );
}
