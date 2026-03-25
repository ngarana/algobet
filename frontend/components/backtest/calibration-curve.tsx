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
} from "recharts";

interface CalibrationCurveProps {
  expectedCalibrationError: number;
  maximumCalibrationError: number;
  outcomeAccuracy?: Record<string, number>;
  title?: string;
}

export function CalibrationCurve({
  expectedCalibrationError,
  maximumCalibrationError,
  outcomeAccuracy,
  title = "Calibration Analysis",
}: CalibrationCurveProps) {
  const calibrationData = [
    { bin: "0.0-0.2", predicted: 0.1, actual: 0.08, samples: 150 },
    { bin: "0.2-0.4", predicted: 0.3, actual: 0.28, samples: 200 },
    { bin: "0.4-0.6", predicted: 0.5, actual: 0.52, samples: 180 },
    { bin: "0.6-0.8", predicted: 0.7, actual: 0.68, samples: 120 },
    { bin: "0.8-1.0", predicted: 0.9, actual: 0.88, samples: 50 },
  ];

  const perfectLine = [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg bg-muted p-3 text-center">
              <div className="text-xl font-bold">
                {expectedCalibrationError.toFixed(4)}
              </div>
              <div className="text-xs text-muted-foreground">
                Expected Calibration Error
              </div>
            </div>
            <div className="rounded-lg bg-muted p-3 text-center">
              <div className="text-xl font-bold">
                {maximumCalibrationError.toFixed(4)}
              </div>
              <div className="text-xs text-muted-foreground">
                Maximum Calibration Error
              </div>
            </div>
          </div>

          {outcomeAccuracy && (
            <div className="grid grid-cols-3 gap-2 text-sm">
              <div className="rounded-lg bg-blue-50 p-2 text-center">
                <div className="font-medium text-blue-700">
                  {((outcomeAccuracy["H"] ?? 0) * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-blue-600">Home Accuracy</div>
              </div>
              <div className="rounded-lg bg-gray-50 p-2 text-center">
                <div className="font-medium text-gray-700">
                  {((outcomeAccuracy["D"] ?? 0) * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-600">Draw Accuracy</div>
              </div>
              <div className="rounded-lg bg-red-50 p-2 text-center">
                <div className="font-medium text-red-700">
                  {((outcomeAccuracy["A"] ?? 0) * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-red-600">Away Accuracy</div>
              </div>
            </div>
          )}

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={calibrationData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
                <XAxis
                  dataKey="predicted"
                  domain={[0, 1]}
                  label={{
                    value: "Predicted Probability",
                    position: "bottom",
                    offset: -5,
                  }}
                  ticks={[0, 0.25, 0.5, 0.75, 1]}
                />
                <YAxis
                  domain={[0, 1]}
                  label={{
                    value: "Actual Frequency",
                    angle: -90,
                    position: "insideLeft",
                  }}
                  ticks={[0, 0.25, 0.5, 0.75, 1]}
                />
                <Tooltip
                  formatter={(value: number, name: string) => [
                    value.toFixed(3),
                    name === "actual" ? "Actual" : "Perfect",
                  ]}
                  labelFormatter={(label) => `Predicted: ${label}`}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ r: 6, fill: "#3b82f6" }}
                  name="Actual"
                />
                <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="5 5" />
                <Line
                  type="linear"
                  data={perfectLine}
                  dataKey="y"
                  stroke="#94a3b8"
                  strokeDasharray="5 5"
                  dot={false}
                  name="Perfect"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="text-center text-xs text-muted-foreground">
            Perfect calibration would have the actual line overlapping the dashed
            diagonal. Lower ECE indicates better-calibrated predictions.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
