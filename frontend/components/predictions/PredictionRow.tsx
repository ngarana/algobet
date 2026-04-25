import { ConfidenceIndicator } from "./ConfidenceIndicator";
import ValueBetIndicator from "./ValueBetIndicator";
import type { Prediction } from "@/lib/types/api";

interface PredictionRowProps {
  prediction: Prediction;
  showRoi: boolean;
}

const outcomeLabels: Record<string, string> = {
  H: "Home Win",
  D: "Draw",
  A: "Away Win",
};

const outcomeBadgeClass: Record<string, string> = {
  H: "bg-blue-600",
  D: "bg-amber-500 text-black",
  A: "bg-red-600",
};

function ProbabilityBar({
  home,
  draw,
  away,
}: {
  home: number;
  draw: number;
  away: number;
}) {
  return (
    <div className="flex h-5 w-full overflow-hidden rounded bg-muted">
      <div
        className="h-full bg-blue-500 transition-all"
        style={{ width: `${home * 100}%` }}
        title={`Home: ${(home * 100).toFixed(1)}%`}
      />
      <div
        className="h-full bg-amber-500 transition-all"
        style={{ width: `${draw * 100}%` }}
        title={`Draw: ${(draw * 100).toFixed(1)}%`}
      />
      <div
        className="h-full bg-red-500 transition-all"
        style={{ width: `${away * 100}%` }}
        title={`Away: ${(away * 100).toFixed(1)}%`}
      />
    </div>
  );
}

export default function PredictionRow({ prediction, showRoi }: PredictionRowProps) {
  const match = prediction.match;
  const matchLabel = match
    ? `${match.home_team_name} vs ${match.away_team_name}`
    : `Match #${prediction.match_id}`;

  const isValueBet = prediction.actual_roi !== null && prediction.actual_roi > 0;

  return (
    <tr className="border-b transition-colors hover:bg-muted/50">
      <td className="p-4 align-middle">
        <div className="font-medium">{matchLabel}</div>
        {match?.tournament_name && (
          <div className="text-xs text-muted-foreground">
            {match.tournament_name}
            {match.season_name ? ` • ${match.season_name}` : ""}
          </div>
        )}
        <div className="mt-2 md:hidden">
          <ProbabilityBar
            home={prediction.prob_home}
            draw={prediction.prob_draw}
            away={prediction.prob_away}
          />
        </div>
      </td>
      <td className="p-4 align-middle">
        <div className="flex flex-col gap-1">
          <span
            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${outcomeBadgeClass[prediction.predicted_outcome]}`}
          >
            {outcomeLabels[prediction.predicted_outcome]}
          </span>
          {prediction.model_version && (
            <span className="text-xs text-muted-foreground">
              {prediction.model_version.version}
            </span>
          )}
        </div>
      </td>
      <td className="p-4 align-middle">
        <div className="hidden md:block">
          <ProbabilityBar
            home={prediction.prob_home}
            draw={prediction.prob_draw}
            away={prediction.prob_away}
          />
        </div>
        <div className="font-mono text-xs md:hidden">
          {(prediction.prob_home * 100).toFixed(1)} /
          {(prediction.prob_draw * 100).toFixed(1)} /
          {(prediction.prob_away * 100).toFixed(1)}
        </div>
      </td>
      <td className="p-4 align-middle">
        <div className="flex items-center gap-2">
          <ConfidenceIndicator confidence={prediction.confidence} size="sm" />
          <span className="hidden font-mono md:inline">
            {(prediction.confidence * 100).toFixed(1)}%
          </span>
        </div>
      </td>
      <td className="p-4 align-middle">
        <div>{match ? new Date(match.match_date).toLocaleString() : "-"}</div>
        <div className="text-xs text-muted-foreground">
          Generated {new Date(prediction.predicted_at).toLocaleString()}
        </div>
      </td>
      {showRoi && (
        <td
          className={`p-4 align-middle font-mono ${
            prediction.actual_roi !== null
              ? prediction.actual_roi >= 0
                ? "text-green-600"
                : "text-red-600"
              : "text-muted-foreground"
          }`}
        >
          {prediction.actual_roi !== null ? (
            <>
              {prediction.actual_roi >= 0 ? "+" : ""}
              {(prediction.actual_roi * 100).toFixed(1)}%
              {isValueBet && (
                <span className="ml-1">
                  <ValueBetIndicator expectedValue={prediction.actual_roi} />
                </span>
              )}
            </>
          ) : (
            "-"
          )}
        </td>
      )}
    </tr>
  );
}
