/**
 * Classification Report Table Component
 * Sortable table showing per-class precision, recall, F1-score, and support.
 * @module
 */

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";

interface ClassMetrics {
  precision: number;
  recall: number;
  "f1-score": number;
  support: number;
}

interface ClassificationReportTableProps {
  classificationReport: Record<string, ClassMetrics>;
  classNames: string[];
}

type SortField = "class" | "precision" | "recall" | "f1-score" | "support";
type SortDirection = "asc" | "desc";

export default function ClassificationReportTable({
  classificationReport,
  classNames,
}: ClassificationReportTableProps) {
  const [sortField, setSortField] = useState<SortField>("f1-score");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  // Extract per-class metrics
  const classMetrics = useMemo(() => {
    return classNames.map((className) => ({
      class: className,
      ...classificationReport[className],
    }));
  }, [classNames, classificationReport]);

  // Extract averages
  const macroAvg = classificationReport["macro avg"];
  const weightedAvg = classificationReport["weighted avg"];

  // Sorted class metrics
  const sortedMetrics = useMemo(() => {
    const sorted = [...classMetrics];
    sorted.sort((a, b) => {
      let aVal: number | string = a[sortField];
      let bVal: number | string = b[sortField];

      if (typeof aVal === "string") {
        return sortDirection === "asc" ? aVal.localeCompare(bVal as string) : (bVal as string).localeCompare(aVal);
      }

      const diff = (aVal as number) - (bVal as number);
      return sortDirection === "asc" ? diff : -diff;
    });
    return sorted;
  }, [classMetrics, sortField, sortDirection]);

  // Find best and worst F1 scores
  const f1Scores = classMetrics.map((m) => m["f1-score"]);
  const bestF1 = Math.max(...f1Scores);
  const worstF1 = Math.min(...f1Scores);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <ArrowUpDown className="w-4 h-4 ml-1 inline opacity-40" />;
    return sortDirection === "asc" ? (
      <ArrowUp className="w-4 h-4 ml-1 inline" />
    ) : (
      <ArrowDown className="w-4 h-4 ml-1 inline" />
    );
  };

  // F1 bar chart (mini horizontal bars)
  const F1Bar = ({ value }: { value: number }) => {
    const widthPercent = (value * 100).toFixed(0);
    return (
      <div className="flex items-center gap-2">
        <span className="text-sm w-12">{value.toFixed(3)}</span>
        <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-[100px]">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all"
            style={{ width: `${widthPercent}%` }}
          />
        </div>
      </div>
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Classification Report</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th
                  className="text-left p-2 cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSort("class")}
                >
                  Class
                  <SortIcon field="class" />
                </th>
                <th
                  className="text-right p-2 cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSort("precision")}
                >
                  Precision
                  <SortIcon field="precision" />
                </th>
                <th
                  className="text-right p-2 cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSort("recall")}
                >
                  Recall
                  <SortIcon field="recall" />
                </th>
                <th
                  className="text-right p-2 cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSort("f1-score")}
                >
                  F1-Score
                  <SortIcon field="f1-score" />
                </th>
                <th
                  className="text-right p-2 cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSort("support")}
                >
                  Support
                  <SortIcon field="support" />
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedMetrics.map((metrics) => {
                const isBest = metrics["f1-score"] === bestF1;
                const isWorst = metrics["f1-score"] === worstF1 && classMetrics.length > 1;
                const bgClass = isBest ? "bg-green-50" : isWorst ? "bg-red-50" : "";

                return (
                  <tr key={metrics.class} className={`border-b hover:bg-gray-50 ${bgClass}`}>
                    <td className="p-2 font-medium">{metrics.class}</td>
                    <td className="p-2 text-right">{metrics.precision.toFixed(3)}</td>
                    <td className="p-2 text-right">{metrics.recall.toFixed(3)}</td>
                    <td className="p-2 text-right">
                      <F1Bar value={metrics["f1-score"]} />
                    </td>
                    <td className="p-2 text-right">{metrics.support}</td>
                  </tr>
                );
              })}

              {/* Macro Average */}
              {macroAvg && (
                <tr className="border-b bg-blue-50 font-semibold">
                  <td className="p-2">Macro Avg</td>
                  <td className="p-2 text-right">{macroAvg.precision.toFixed(3)}</td>
                  <td className="p-2 text-right">{macroAvg.recall.toFixed(3)}</td>
                  <td className="p-2 text-right">{macroAvg["f1-score"].toFixed(3)}</td>
                  <td className="p-2 text-right">{macroAvg.support}</td>
                </tr>
              )}

              {/* Weighted Average */}
              {weightedAvg && (
                <tr className="bg-blue-100 font-semibold">
                  <td className="p-2">Weighted Avg</td>
                  <td className="p-2 text-right">{weightedAvg.precision.toFixed(3)}</td>
                  <td className="p-2 text-right">{weightedAvg.recall.toFixed(3)}</td>
                  <td className="p-2 text-right">{weightedAvg["f1-score"].toFixed(3)}</td>
                  <td className="p-2 text-right">{weightedAvg.support}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
