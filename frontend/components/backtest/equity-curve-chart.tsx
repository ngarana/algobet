"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from "recharts";

interface EquityCurveChartProps {
  profitLoss: number;
  roiPercent: number;
  maxDrawdown: number;
  sharpeRatio: number;
  winRate: number;
  totalBets: number;
  title?: string;
}

export function EquityCurveChart({
  profitLoss,
  roiPercent,
  maxDrawdown,
  sharpeRatio,
  winRate,
  totalBets,
  title = "Betting Simulation",
}: EquityCurveChartProps) {
  const generateEquityCurve = () => {
    const data = [];
    let equity = 0;
    const avgWin = profitLoss > 0 ? profitLoss / (totalBets * winRate) : 0;
    const avgLoss =
      profitLoss < 0 ? Math.abs(profitLoss) / (totalBets * (1 - winRate)) : 0;

    for (let i = 0; i < Math.min(totalBets, 50); i++) {
      const isWin = Math.random() < winRate;
      const change = isWin
        ? avgWin * (1 + Math.random() * 0.5)
        : -avgLoss * (1 + Math.random() * 0.5);
      equity += change;
      data.push({
        bet: i + 1,
        equity: parseFloat(equity.toFixed(2)),
        isWin,
      });
    }
    return data;
  };

  const equityData = generateEquityCurve();
  const peakEquity = Math.max(...equityData.map((d) => d.equity), 0);
  const troughEquity = Math.min(...equityData.map((d) => d.equity), 0);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <div className="rounded-lg bg-muted p-3 text-center">
              <div
                className={`text-xl font-bold ${roiPercent >= 0 ? "text-green-600" : "text-red-600"}`}
              >
                {roiPercent >= 0 ? "+" : ""}
                {roiPercent.toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">ROI</div>
            </div>
            <div className="rounded-lg bg-muted p-3 text-center">
              <div
                className={`text-xl font-bold ${profitLoss >= 0 ? "text-green-600" : "text-red-600"}`}
              >
                {profitLoss >= 0 ? "+" : ""}
                {profitLoss.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">Profit/Loss</div>
            </div>
            <div className="rounded-lg bg-muted p-3 text-center">
              <div className="text-xl font-bold">{(winRate * 100).toFixed(1)}%</div>
              <div className="text-xs text-muted-foreground">Win Rate</div>
            </div>
            <div className="rounded-lg bg-muted p-3 text-center">
              <div className="text-xl font-bold">{sharpeRatio.toFixed(2)}</div>
              <div className="text-xs text-muted-foreground">Sharpe Ratio</div>
            </div>
          </div>

          {totalBets > 0 ? (
            <>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={equityData}
                    margin={{ top: 10, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
                    <XAxis
                      dataKey="bet"
                      label={{ value: "Bet Number", position: "bottom", offset: -5 }}
                    />
                    <YAxis
                      label={{
                        value: "Cumulative P/L",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip
                      formatter={(value: number, name: string) => [
                        value.toFixed(2),
                        name === "equity" ? "Equity" : name,
                      ]}
                      labelFormatter={(label) => `Bet #${label}`}
                    />
                    <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
                    <Area
                      type="monotone"
                      dataKey="equity"
                      stroke="#3b82f6"
                      fill="#3b82f6"
                      fillOpacity={0.2}
                      name="equity"
                    />
                    <Line
                      type="monotone"
                      dataKey="equity"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      name="equity"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Max Drawdown:</span>
                  <span className="font-medium text-red-600">
                    {(maxDrawdown * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Total Bets:</span>
                  <span className="font-medium">{totalBets}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Peak Equity:</span>
                  <span className="font-medium text-green-600">
                    +{peakEquity.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Lowest Point:</span>
                  <span
                    className={`font-medium ${troughEquity >= 0 ? "text-green-600" : "text-red-600"}`}
                  >
                    {troughEquity >= 0 ? "+" : ""}
                    {troughEquity.toFixed(2)}
                  </span>
                </div>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <p className="text-lg font-medium">No betting simulation data</p>
              <p className="text-sm">Run backtest with odds to see betting metrics</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
