interface ValueBetIndicatorProps {
  expectedValue?: number;
  probHome?: number;
  probDraw?: number;
  probAway?: number;
  oddsHome?: number | null;
  oddsDraw?: number | null;
  oddsAway?: number | null;
  predictedOutcome?: "H" | "D" | "A";
  compact?: boolean;
}

function calculateExpectedValue(probability: number, odds: number): number {
  return probability * odds - 1;
}

export default function ValueBetIndicator({
  expectedValue,
  probHome = 0,
  probDraw = 0,
  probAway = 0,
  oddsHome,
  oddsDraw,
  oddsAway,
  predictedOutcome,
  compact = false,
}: ValueBetIndicatorProps) {
  if (expectedValue === undefined && predictedOutcome) {
    const probability =
      predictedOutcome === "H"
        ? probHome
        : predictedOutcome === "D"
          ? probDraw
          : probAway;

    const odds =
      predictedOutcome === "H"
        ? oddsHome
        : predictedOutcome === "D"
          ? oddsDraw
          : oddsAway;

    if (odds === undefined || odds === null || probability === undefined) {
      return null;
    }

    const calculatedEV = calculateExpectedValue(probability, odds);
    const isValueBet = calculatedEV > 0;

    if (compact) {
      return (
        <span
          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${
            isValueBet ? "bg-green-500 text-white" : "bg-gray-200 text-gray-600"
          }`}
        >
          {isValueBet ? "Value" : "No Value"}
        </span>
      );
    }

    return (
      <span
        className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${
          isValueBet ? "bg-green-500 text-white" : "bg-gray-200 text-gray-600"
        }`}
      >
        {isValueBet ? "+" : ""}
        {(calculatedEV * 100).toFixed(1)}% EV
      </span>
    );
  }

  if (expectedValue === undefined) {
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
