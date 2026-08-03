/**
 * Confusion Matrix Panel Component
 * Renders confusion matrix as a heatmap grid with hover tooltips.
 * @module
 */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface ConfusionMatrixPanelProps {
  confusionMatrix: number[][];
  classNames: string[];
  accuracy: number;
}

export default function ConfusionMatrixPanel({
  confusionMatrix,
  classNames,
  accuracy,
}: ConfusionMatrixPanelProps) {
  const numClasses = classNames.length;

  // Find max value for color scaling
  const maxValue = Math.max(...confusionMatrix.flat());

  // Color intensity function
  const getColor = (value: number, isDiagonal: boolean) => {
    const intensity = maxValue > 0 ? value / maxValue : 0;
    if (isDiagonal) {
      // Green scale for correct predictions (diagonal)
      const greenValue = Math.floor(200 - intensity * 100);
      return `rgb(${greenValue}, ${200 + intensity * 55}, ${greenValue})`;
    } else {
      // Blue scale for off-diagonal (errors)
      const blueValue = Math.floor(220 - intensity * 120);
      return `rgb(${blueValue}, ${blueValue}, ${220 + intensity * 35})`;
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Confusion Matrix</CardTitle>
          <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
            Accuracy: {(accuracy * 100).toFixed(2)}%
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* Matrix Grid */}
            <div className="grid gap-0" style={{ gridTemplateColumns: `auto repeat(${numClasses}, 1fr)` }}>
              {/* Top-left corner (empty) */}
              <div className="p-2 text-xs font-medium text-gray-500"></div>

              {/* Column headers (Predicted) */}
              {classNames.map((name, idx) => (
                <div key={`col-${idx}`} className="p-2 text-xs font-medium text-center text-gray-700">
                  {name}
                </div>
              ))}

              {/* Row headers + cells */}
              {confusionMatrix.map((row, rowIdx) => (
                <div key={`row-${rowIdx}`} className="contents">
                  {/* Row header (Actual) */}
                  <div className="p-2 text-xs font-medium text-gray-700 flex items-center">
                    {classNames[rowIdx]}
                  </div>

                  {/* Cells */}
                  {row.map((value, colIdx) => {
                    const isDiagonal = rowIdx === colIdx;
                    const color = getColor(value, isDiagonal);

                    return (
                      <div
                        key={`cell-${rowIdx}-${colIdx}`}
                        className="relative group p-4 border border-gray-200 text-center cursor-pointer transition-all hover:ring-2 hover:ring-blue-400"
                        style={{ backgroundColor: color }}
                      >
                        <span className="text-sm font-semibold text-gray-800">{value}</span>

                        {/* Hover Tooltip */}
                        <div className="absolute hidden group-hover:block bg-gray-900 text-white text-xs rounded px-2 py-1 z-10 -top-8 left-1/2 transform -translate-x-1/2 whitespace-nowrap">
                          Predicted: {classNames[colIdx]} | Actual: {classNames[rowIdx]} | Count: {value}
                          <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>

            {/* Axis Labels */}
            <div className="mt-4 flex justify-between text-xs text-gray-500">
              <div>
                <span className="font-medium">Rows:</span> Actual Class
              </div>
              <div>
                <span className="font-medium">Columns:</span> Predicted Class
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
