/**
 * Loss Chart Component
 * Displays training and validation loss curves using Recharts.
 * @module
 */

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ChartProps } from '@/types/training';

export default function LossChart({ metrics }: ChartProps) {
  // Check if we have validation data
  const hasValidation = metrics.some(m => m.val_loss != null);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Loss</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={metrics}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="epoch" 
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip 
              formatter={(value: number) => value.toFixed(4)}
              labelFormatter={(label) => `Epoch ${label}`}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#3b82f6"
              strokeWidth={2}
              name="Training Loss"
              dot={false}
              isAnimationActive={false}
            />
            {hasValidation && (
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke="#f97316"
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Validation Loss"
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
