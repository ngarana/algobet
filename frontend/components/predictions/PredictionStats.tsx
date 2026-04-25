interface PredictionStatsProps {
  predictions: Array<{
    confidence: number;
    predicted_outcome: string;
    actual_roi?: number | null;
  }>;
  filteredPredictions?: Array<{
    confidence: number;
    predicted_outcome: string;
    actual_roi?: number | null;
  }>;
}

export default function PredictionStats({
  predictions,
  filteredPredictions,
}: PredictionStatsProps) {
  const displayPredictions = filteredPredictions || predictions;

  const avgConfidence =
    displayPredictions.length > 0
      ? displayPredictions.reduce((sum, prediction) => sum + prediction.confidence, 0) /
        displayPredictions.length
      : 0;

  const counts = displayPredictions.reduce(
    (accumulator, prediction) => {
      accumulator[prediction.predicted_outcome] =
        (accumulator[prediction.predicted_outcome] || 0) + 1;
      return accumulator;
    },
    {} as Record<string, number>
  );

  return (
    <div className="grid gap-4 md:grid-cols-4">
      <div className="rounded-lg border bg-card p-4 text-center">
        <div className="text-2xl font-bold">{displayPredictions.length}</div>
        <div className="text-xs text-muted-foreground">Predictions</div>
      </div>
      <div className="rounded-lg border bg-card p-4 text-center">
        <div className="text-2xl font-bold">{(avgConfidence * 100).toFixed(1)}%</div>
        <div className="text-xs text-muted-foreground">Avg Confidence</div>
      </div>
      <div className="rounded-lg border bg-card p-4 text-center">
        <div className="text-2xl font-bold text-blue-600">{counts.H ?? 0}</div>
        <div className="text-xs text-muted-foreground">Home Picks</div>
      </div>
      <div className="rounded-lg border bg-card p-4 text-center">
        <div className="text-2xl font-bold text-red-600">{counts.A ?? 0}</div>
        <div className="text-xs text-muted-foreground">Away Picks</div>
      </div>
    </div>
  );
}
