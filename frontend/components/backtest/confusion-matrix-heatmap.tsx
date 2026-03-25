"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ConfusionMatrixHeatmapProps {
  confusionMatrix: number[][];
  labels?: string[];
  title?: string;
}

const OUTCOME_COLORS = {
  H: { bg: "bg-blue-100", border: "border-blue-300", text: "text-blue-800" },
  D: { bg: "bg-gray-100", border: "border-gray-300", text: "text-gray-800" },
  A: { bg: "bg-red-100", border: "border-red-300", text: "text-red-800" },
};

export function ConfusionMatrixHeatmap({
  confusionMatrix,
  labels = ["H", "D", "A"],
  title = "Confusion Matrix",
}: ConfusionMatrixHeatmapProps) {
  if (!confusionMatrix || confusionMatrix.length !== 3) {
    return null;
  }

  const total = confusionMatrix.flat().reduce((a, b) => a + b, 0);
  const maxValue = Math.max(...confusionMatrix.flat());

  const getColor = (value: number, maxValue: number) => {
    const intensity = value / maxValue;
    if (intensity > 0.7) return "bg-green-600 text-white";
    if (intensity > 0.4) return "bg-green-400 text-green-900";
    if (intensity > 0.2) return "bg-green-200 text-green-800";
    if (intensity > 0.1) return "bg-green-100 text-green-700";
    return "bg-gray-50 text-gray-600";
  };

  const getPercentage = (value: number) => {
    if (total === 0) return 0;
    return (value / total) * 100;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center">
          <div className="mb-2 text-sm text-muted-foreground">Predicted Outcome</div>
          <div className="flex items-center">
            <div
              className="mr-2 flex flex-col justify-around text-sm text-muted-foreground"
              style={{ height: "180px" }}
            >
              <div className="flex items-center">
                <span className="writing-mode-vertical">Actual</span>
              </div>
            </div>
            <div>
              <div className="mb-1 flex justify-center gap-1">
                {labels.map((label) => (
                  <div key={label} className="w-20 text-center text-sm font-medium">
                    {label === "H" ? "Home" : label === "D" ? "Draw" : "Away"}
                  </div>
                ))}
              </div>
              {confusionMatrix.map((row, actualIdx) => (
                <div key={actualIdx} className="mb-1 flex gap-1">
                  <div className="w-12 pr-2 text-right text-sm font-medium">
                    {labels[actualIdx] === "H"
                      ? "Home"
                      : labels[actualIdx] === "D"
                        ? "Draw"
                        : "Away"}
                  </div>
                  {row.map((value, predictedIdx) => (
                    <div
                      key={predictedIdx}
                      className={`flex h-14 w-20 flex-col items-center justify-center rounded-md border transition-colors ${getColor(value, maxValue)}`}
                      title={`${value} samples (${getPercentage(value).toFixed(1)}%)`}
                    >
                      <span className="text-lg font-bold">{value}</span>
                      <span className="text-xs opacity-80">
                        {getPercentage(value).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
          <div className="mt-4 flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="h-4 w-4 rounded bg-green-600" />
              <span>High</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-4 w-4 rounded bg-green-200" />
              <span>Medium</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-4 w-4 rounded bg-gray-100" />
              <span>Low</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
