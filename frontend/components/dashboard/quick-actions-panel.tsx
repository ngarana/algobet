"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Target,
  TrendingUp,
  Brain,
  Calculator,
  RotateCcw,
  Calendar,
  Trophy,
} from "lucide-react";

interface QuickActionsPanelProps {
  onRunPredictions?: () => void;
  onViewValueBets?: () => void;
  onCalibrateModel?: () => void;
  onBacktestModel?: () => void;
}

export function QuickActionsPanel({
  onRunPredictions,
  onViewValueBets,
  onCalibrateModel,
  onBacktestModel,
}: QuickActionsPanelProps) {
  return (
    <div className="grid grid-cols-2 gap-2">
      <Button
        variant="outline"
        className="h-auto justify-start py-3"
        onClick={onRunPredictions}
      >
        <Target className="mr-2 h-4 w-4" />
        <div className="text-left">
          <div>Run Predictions</div>
          <div className="text-xs text-muted-foreground">Generate new predictions</div>
        </div>
      </Button>

      <Button
        variant="outline"
        className="h-auto justify-start py-3"
        onClick={onViewValueBets}
      >
        <TrendingUp className="mr-2 h-4 w-4" />
        <div className="text-left">
          <div>Value Bets</div>
          <div className="text-xs text-muted-foreground">View opportunities</div>
        </div>
      </Button>

      <Button
        variant="outline"
        className="h-auto justify-start py-3"
        onClick={onCalibrateModel}
      >
        <Calculator className="mr-2 h-4 w-4" />
        <div className="text-left">
          <div>Calibrate Model</div>
          <div className="text-xs text-muted-foreground">Adjust model parameters</div>
        </div>
      </Button>

      <Button
        variant="outline"
        className="h-auto justify-start py-3"
        onClick={onBacktestModel}
      >
        <RotateCcw className="mr-2 h-4 w-4" />
        <div className="text-left">
          <div>Backtest Model</div>
          <div className="text-xs text-muted-foreground">Test historical data</div>
        </div>
      </Button>
    </div>
  );
}
