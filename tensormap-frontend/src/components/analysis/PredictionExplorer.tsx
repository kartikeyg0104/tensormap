/**
 * Prediction Explorer Component
 * Paginated table showing predictions with filtering and expandable rows.
 * @module
 */

import { useState, useEffect, Fragment } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { ChevronLeft, ChevronRight, ChevronDown, ChevronUp, Check, X } from "lucide-react";
import axios from "@/shared/Axios";
import { BACKEND_MODEL_ANALYSIS } from "@/constants/Urls";
import { Prediction, PredictionsData } from "@/types/analysis";
import logger from "@/shared/logger";

interface PredictionExplorerProps {
  jobId: string;
}

export default function PredictionExplorer({ jobId }: PredictionExplorerProps) {
  const [data, setData] = useState<PredictionsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "correct" | "incorrect">("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [sortByConfidence, setSortByConfidence] = useState<"desc" | "asc">("desc");

  const ITEMS_PER_PAGE = 25;

  // Fetch predictions
  const fetchPredictions = async (offset: number, filterValue: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const filterParam = filterValue === "all" ? undefined : filterValue;
      const response = await axios.get(`${BACKEND_MODEL_ANALYSIS}/${jobId}/predictions`, {
        params: {
          offset,
          limit: ITEMS_PER_PAGE,
          filter: filterParam,
        },
      });

      setData(response.data as PredictionsData);
    } catch (err: any) {
      logger.error("Failed to fetch predictions:", err);
      setError(err.response?.data?.detail || err.message || "Failed to load predictions");
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch on mount and when page/filter changes
  useEffect(() => {
    const offset = (currentPage - 1) * ITEMS_PER_PAGE;
    fetchPredictions(offset, filter);
  }, [jobId, currentPage, filter]);

  const totalPages = data ? Math.ceil(data.total / ITEMS_PER_PAGE) : 0;

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      setExpandedRow(null); // Collapse expanded row on page change
    }
  };

  const toggleRow = (index: number) => {
    setExpandedRow(expandedRow === index ? null : index);
  };

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Prediction Explorer</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-800">
              <strong>Error:</strong> {error}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Prediction Explorer</CardTitle>
          <div className="flex items-center gap-2">
            <Select
              value={filter}
              onValueChange={(val) => {
                setFilter(val as typeof filter);
                setCurrentPage(1); // Reset pagination when filter changes
                setExpandedRow(null); // Collapse expanded row
              }}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Filter predictions" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Predictions</SelectItem>
                <SelectItem value="correct">Correct Only</SelectItem>
                <SelectItem value="incorrect">Incorrect Only</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : data && data.predictions && data.predictions.length > 0 ? (
          <>
            {/* Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-gray-50">
                    <th className="text-left p-2 w-12">#</th>
                    <th className="text-left p-2">Actual Label</th>
                    <th className="text-left p-2">Predicted Label</th>
                    <th
                      className="text-right p-2 cursor-pointer hover:bg-gray-100"
                      onClick={() => setSortByConfidence(sortByConfidence === "desc" ? "asc" : "desc")}
                    >
                      Confidence {sortByConfidence === "desc" ? "↓" : "↑"}
                    </th>
                    <th className="text-center p-2">Status</th>
                    <th className="text-center p-2 w-12"></th>
                  </tr>
                </thead>
                <tbody>
                  {[...data.predictions]
                    .sort((a, b) =>
                      sortByConfidence === "desc"
                        ? b.confidence - a.confidence
                        : a.confidence - b.confidence
                    )
                    .map((pred: Prediction) => (
                      <Fragment key={pred.index}>
                        <tr
                          className="border-b hover:bg-gray-50 cursor-pointer"
                          onClick={() => toggleRow(pred.index)}
                        >
                          <td className="p-2 text-gray-500">{pred.index}</td>
                          <td className="p-2 font-medium">{pred.actual_class_name}</td>
                          <td className="p-2 font-medium">{pred.predicted_class_name}</td>
                          <td className="p-2 text-right">{(pred.confidence * 100).toFixed(2)}%</td>
                          <td className="p-2 text-center">
                            {pred.is_correct ? (
                              <Badge className="bg-green-100 text-green-700 border-green-200">
                                <Check className="w-3 h-3 mr-1" />
                                Correct
                              </Badge>
                            ) : (
                              <Badge className="bg-red-100 text-red-700 border-red-200">
                                <X className="w-3 h-3 mr-1" />
                                Misclassified
                              </Badge>
                            )}
                          </td>
                          <td className="p-2 text-center">
                            {expandedRow === pred.index ? (
                              <ChevronUp className="w-4 h-4 mx-auto" />
                            ) : (
                              <ChevronDown className="w-4 h-4 mx-auto" />
                            )}
                          </td>
                        </tr>

                        {/* Expanded Row */}
                        {expandedRow === pred.index && (
                          <tr className="bg-blue-50">
                            <td colSpan={6} className="p-4">
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {/* Probability Breakdown */}
                                <div>
                                  <h4 className="text-sm font-semibold mb-2">Probability Breakdown</h4>
                                  <div className="space-y-1">
                                    {pred.probabilities.map((prob, idx) => (
                                      <div key={idx} className="flex items-center gap-2">
                                        <span className="text-xs w-16">Class {idx}:</span>
                                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                                          <div
                                            className="bg-blue-500 h-2 rounded-full"
                                            style={{ width: `${(prob * 100).toFixed(0)}%` }}
                                          />
                                        </div>
                                        <span className="text-xs w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>

                                {/* Features */}
                                <div>
                                  <h4 className="text-sm font-semibold mb-2">Feature Values</h4>
                                  <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto text-xs">
                                    {Object.entries(pred.features).map(([key, value]) => (
                                      <div key={key} className="flex justify-between">
                                        <span className="text-gray-600">{key}:</span>
                                        <span className="font-mono">{value.toFixed(3)}</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="mt-4 flex items-center justify-between">
              <div className="text-sm text-gray-600">
                Showing {data.offset + 1} - {Math.min(data.offset + ITEMS_PER_PAGE, data.total)} of {data.total}
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>

                {/* Page numbers */}
                <div className="flex items-center gap-1">
                  {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                    let pageNum: number;
                    if (totalPages <= 5) {
                      pageNum = i + 1;
                    } else if (currentPage <= 3) {
                      pageNum = i + 1;
                    } else if (currentPage >= totalPages - 2) {
                      pageNum = totalPages - 4 + i;
                    } else {
                      pageNum = currentPage - 2 + i;
                    }

                    return (
                      <Button
                        key={pageNum}
                        variant={currentPage === pageNum ? "default" : "outline"}
                        size="sm"
                        onClick={() => handlePageChange(pageNum)}
                      >
                        {pageNum}
                      </Button>
                    );
                  })}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages}
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-12 text-gray-500">
            <p>No predictions found.</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
