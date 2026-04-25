import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfidenceIndicator } from "@/components/predictions/ConfidenceIndicator";
import ValueBetIndicator from "@/components/predictions/ValueBetIndicator";
import { TrendingUp, Calendar, ChevronDown, ChevronUp, Eye } from "lucide-react";
import { useState } from "react";
import type { Prediction } from "@/lib/types/api";

interface PredictionCardProps {
  prediction: Prediction;
  showRoi?: boolean;
  onViewDetails?: (prediction: Prediction) => void;
  compact?: boolean;
}

function getOutcomeLabel(outcome: string): string {
  switch (outcome) {
    case "H":
      return "Home";
    case "D":
      return "Draw";
    case "A":
      return "Away";
    default:
      return outcome;
  }
}

function getOutcomeColor(outcome: string): string {
  switch (outcome) {
    case "H":
      return "text-blue-600 bg-blue-50";
    case "D":
      return "text-amber-600 bg-amber-50";
    case "A":
      return "text-red-600 bg-red-50";
    default:
      return "";
  }
}

export default function PredictionCard({
  prediction,
  showRoi = false,
  onViewDetails,
  compact = false,
}: PredictionCardProps) {
  const [expanded, setExpanded] = useState(false);
  const match = prediction.match as
    | {
        home_team_name: string;
        away_team_name: string;
        tournament_name?: string | null;
        match_date: string;
        status: string;
        home_score?: number | null;
        away_score?: number | null;
        odds_home?: number | null;
        odds_draw?: number | null;
        odds_away?: number | null;
      }
    | null
    | undefined;

  const predictedOutcome = prediction.predicted_outcome;

  return (
    <Card className="overflow-hidden">
      <CardContent className="space-y-3 p-4">
        {match && (
          <>
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">
                {match.home_team_name} vs {match.away_team_name}
              </div>
              {match.tournament_name && (
                <Badge variant="outline" className="text-xs">
                  {match.tournament_name}
                </Badge>
              )}
            </div>

            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Calendar className="h-3 w-3" />
              {new Date(match.match_date).toLocaleDateString()}
              {match.status === "FINISHED" && match.home_score !== null && (
                <Badge variant="secondary" className="ml-2 text-xs">
                  {match.home_score} - {match.away_score}
                </Badge>
              )}
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-xs text-muted-foreground">H</div>
                <div className="text-sm font-bold text-blue-600">
                  {(prediction.prob_home * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">D</div>
                <div className="text-sm font-bold text-amber-600">
                  {(prediction.prob_draw * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">A</div>
                <div className="text-sm font-bold text-red-600">
                  {(prediction.prob_away * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <Badge className={getOutcomeColor(predictedOutcome)}>
                {getOutcomeLabel(predictedOutcome)}
              </Badge>
              <ConfidenceIndicator confidence={prediction.confidence} size="sm" />
            </div>

            {!compact && (
              <>
                <ValueBetIndicator
                  probHome={prediction.prob_home}
                  probDraw={prediction.prob_draw}
                  probAway={prediction.prob_away}
                  oddsHome={match.odds_home ?? null}
                  oddsDraw={match.odds_draw ?? null}
                  oddsAway={match.odds_away ?? null}
                  predictedOutcome={predictedOutcome}
                  compact
                />

                {showRoi &&
                  prediction.actual_roi !== null &&
                  prediction.actual_roi !== undefined && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-1 text-muted-foreground">
                        <TrendingUp className="h-3 w-3" />
                        ROI
                      </span>
                      <span
                        className={`font-bold ${
                          prediction.actual_roi >= 0 ? "text-green-600" : "text-red-600"
                        }`}
                      >
                        {prediction.actual_roi >= 0 ? "+" : ""}
                        {prediction.actual_roi.toFixed(2)}%
                      </span>
                    </div>
                  )}
              </>
            )}

            <div className="flex items-center justify-between border-t pt-2">
              {!compact && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setExpanded(!expanded)}
                >
                  {expanded ? (
                    <>
                      <ChevronUp className="mr-1 h-3 w-3" />
                      Less
                    </>
                  ) : (
                    <>
                      <ChevronDown className="mr-1 h-3 w-3" />
                      More
                    </>
                  )}
                </Button>
              )}
              {onViewDetails && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onViewDetails(prediction)}
                >
                  <Eye className="mr-1 h-3 w-3" />
                  Details
                </Button>
              )}
            </div>

            {expanded && !compact && (
              <div className="space-y-2 border-t pt-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Max Probability</span>
                  <span>{(prediction.max_probability * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Predicted At</span>
                  <span>{new Date(prediction.predicted_at).toLocaleDateString()}</span>
                </div>
                {match.odds_home && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Best Odds</span>
                    <span>
                      {predictedOutcome === "H"
                        ? match.odds_home
                        : predictedOutcome === "D"
                          ? match.odds_draw
                          : match.odds_away}
                    </span>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
