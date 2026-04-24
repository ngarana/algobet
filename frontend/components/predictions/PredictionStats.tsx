interface PredictionStatsProps {
  predictions: Array<{
    confidence: number;
    predicted_outcome: string;
  }>;
}

export default function PredictionStats({ predictions }: PredictionStatsProps) {
  const avgConfidence =
    predictions.length > 0
      ? predictions.reduce((sum, prediction) => sum + prediction.confidence, 0) /
        predictions.length
      : 0;

  const counts = predictions.reduce(
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
        <div className="text-2xl font-bold">{predictions.length}</div>
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
