interface ValueBetIndicatorProps {
  expectedValue?: number;
  _kellyFraction?: number;
  marketOdds?: number;
  predictedProbability?: number;
}

export default function ValueBetIndicator({
  expectedValue,
  marketOdds,
  predictedProbability,
}: ValueBetIndicatorProps) {
  if (
    expectedValue === undefined ||
    marketOdds === undefined ||
    predictedProbability === undefined
  ) {
    return null;
  }

  const isValueBet = expectedValue > 0;

  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${
        isValueBet ? "bg-green-500 text-white" : "bg-gray-200 text-gray-600"
      }`}
    >
      {isValueBet ? "+" : ""}
      {(expectedValue * 100).toFixed(1)}% EV
    </span>
  );
}
