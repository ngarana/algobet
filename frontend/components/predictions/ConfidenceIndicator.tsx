interface ConfidenceIndicatorProps {
  confidence: number;
  size?: "sm" | "md" | "lg";
}

export function ConfidenceIndicator({
  confidence,
  size = "md",
}: ConfidenceIndicatorProps) {
  const percentage = (confidence * 100).toFixed(1);

  const colorClass =
    confidence >= 0.7
      ? "bg-green-500"
      : confidence >= 0.5
        ? "bg-yellow-500"
        : "bg-red-500";

  const sizeClass =
    size === "sm" ? "h-1.5 w-16" : size === "lg" ? "h-3 w-32" : "h-2 w-24";

  return (
    <div className="flex items-center gap-2">
      <div className={`h-2 w-24 rounded-full bg-muted ${sizeClass}`}>
        <div
          className={`h-full rounded-full ${colorClass}`}
          style={{ width: `${confidence * 100}%` }}
        />
      </div>
      <span className="font-mono text-xs">{percentage}%</span>
    </div>
  );
}
