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

export default function PredictionRow({ prediction, showRoi }: PredictionRowProps) {
  const match = prediction.match;
  const matchLabel = match
    ? `${match.home_team_name} vs ${match.away_team_name}`
    : `Match #${prediction.match_id}`;

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
      <td className="p-4 align-middle font-mono text-xs">
        {(prediction.prob_home * 100).toFixed(1)} /{" "}
        {(prediction.prob_draw * 100).toFixed(1)} /{" "}
        {(prediction.prob_away * 100).toFixed(1)}
      </td>
      <td className="p-4 align-middle font-mono">
        {(prediction.confidence * 100).toFixed(1)}%
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
          {prediction.actual_roi !== null
            ? `${prediction.actual_roi >= 0 ? "+" : ""}${(prediction.actual_roi * 100).toFixed(1)}%`
            : "-"}
        </td>
      )}
    </tr>
  );
}
