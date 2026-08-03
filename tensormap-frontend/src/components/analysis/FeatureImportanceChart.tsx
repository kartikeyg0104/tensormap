/**
 * Feature Importance Chart Component
 * Horizontal bar chart showing feature importance with error bars.
 * @module
 */

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar,
  Cell,
} from "recharts";
import { ChevronDown, ChevronUp } from "lucide-react";

interface FeatureImportanceChartProps {
  features: string[];
  importancesMean: number[];
  importancesStd: number[];
}

export default function FeatureImportanceChart({
  features,
  importancesMean,
  importancesStd,
}: FeatureImportanceChartProps) {
  const [showLowImportance, setShowLowImportance] = useState(false);

  // Combine data
  const allData = useMemo(() => {
    return features
      .map((feature, idx) => ({
        feature,
        importance: importancesMean[idx],
        std: importancesStd[idx],
      }))
      .sort((a, b) => b.importance - a.importance); // Sort DESC by importance
  }, [features, importancesMean, importancesStd]);

  // Split into high and low importance
  const LOW_THRESHOLD = 0.001;
  const highImportanceData = allData.filter((d) => d.importance >= LOW_THRESHOLD);
  const lowImportanceData = allData.filter((d) => d.importance < LOW_THRESHOLD);

  const displayData = showLowImportance ? allData : highImportanceData;

  // Color by magnitude
  const getColor = (importance: number) => {
    if (importance < 0.001) return "#9ca3af"; // gray
    if (importance < 0.01) return "#60a5fa"; // light blue
    return "#3b82f6"; // blue
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Feature Importance</CardTitle>
        <p className="text-xs text-gray-500 mt-1">
          Permutation importance (mean ± std across {importancesMean.length > 0 ? "10 repeats" : "N/A"})
        </p>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={Math.max(300, displayData.length * 30)}>
          <BarChart
            data={displayData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis
              type="category"
              dataKey="feature"
              width={90}
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              formatter={(value: number) => value.toFixed(4)}
              labelFormatter={(label) => `Feature: ${label}`}
            />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {displayData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getColor(entry.importance)} />
              ))}
              <ErrorBar dataKey="std" width={4} strokeWidth={1.5} stroke="#666" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Toggle for low-importance features */}
        {lowImportanceData.length > 0 && (
          <div className="mt-4 flex justify-center">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowLowImportance(!showLowImportance)}
            >
              {showLowImportance ? (
                <>
                  <ChevronUp className="w-4 h-4 mr-1" />
                  Hide {lowImportanceData.length} low-importance features
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4 mr-1" />
                  Show {lowImportanceData.length} low-importance features
                </>
              )}
            </Button>
          </div>
        )}

        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-4 text-xs text-gray-600">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: "#3b82f6" }} />
            <span>High importance (&gt;0.01)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: "#60a5fa" }} />
            <span>Medium importance (0.001-0.01)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: "#9ca3af" }} />
            <span>Low importance (&lt;0.001)</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
