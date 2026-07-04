/**
 * Accuracy Chart Component
 * Displays training and validation accuracy curves using Recharts.
 * @module
 */

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartProps } from "@/types/training";

export default function AccuracyChart({ metrics }: ChartProps) {
  // Check if we have accuracy data
  const hasAccuracy = metrics.some((m) => m.accuracy != null);
  const hasValidation = metrics.some((m) => m.val_accuracy != null);

  // Don't render if no accuracy data
  if (!hasAccuracy && !hasValidation) {
    return null;
  }

  // Format percentage
  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Accuracy</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={metrics} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="epoch"
              label={{ value: "Epoch", position: "insideBottom", offset: -5 }}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={formatPercent}
              label={{ value: "Accuracy", angle: -90, position: "insideLeft" }}
            />
            <Tooltip
              formatter={(value: number) => formatPercent(value)}
              labelFormatter={(label) => `Epoch ${label}`}
            />
            <Legend />
            {hasAccuracy && (
              <Line
                type="monotone"
                dataKey="accuracy"
                stroke="#22c55e"
                strokeWidth={2}
                name="Training Accuracy"
                dot={false}
                isAnimationActive={false}
              />
            )}
            {hasValidation && (
              <Line
                type="monotone"
                dataKey="val_accuracy"
                stroke="#14b8a6"
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Validation Accuracy"
                dot={false}
                isAnimationActive={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
