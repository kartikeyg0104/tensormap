/**
 * Hook for polling feature importance analysis with 202/200 pattern.
 *
 * Polls GET /api/v1/model/analysis/{job_id}/feature-importance every 2 seconds.
 * Stops polling when response is 200 (data ready, not 202).
 *
 * @module
 */

import { useState, useEffect, useRef, useCallback } from "react";
import axios from "../shared/Axios";
import { BACKEND_MODEL_ANALYSIS } from "../constants/Urls";
import { FeatureImportanceData } from "../types/analysis";
import logger from "../shared/logger";

const POLL_INTERVAL_MS = 2000;

interface UseFeatureImportancePollerReturn {
  /** Feature importance data, null while loading or if not yet available. */
  data: FeatureImportanceData | null;
  /** True while waiting for computation to complete. */
  isLoading: boolean;
  /** Error message if request failed. */
  error: string | null;
}

/**
 * Polls for feature importance analysis results.
 *
 * On mount, fires the first GET request. If the backend returns 202
 * (computing), polls every 2 seconds. Once 200 is received, stores the
 * data and stops polling.
 *
 * @param jobId - Training job ID to fetch feature importance for.
 *                Pass null to disable polling.
 */
export function useFeatureImportancePoller(
  jobId: string | null,
): UseFeatureImportancePollerReturn {
  const [data, setData] = useState<FeatureImportanceData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const fetchImportance = useCallback(async () => {
    if (!jobId) return;

    try {
      const response = await axios.get(
        `${BACKEND_MODEL_ANALYSIS}/${jobId}/feature-importance`,
        {
          // Don't throw on 202 — it's expected during computation
          validateStatus: (status: number) => status === 200 || status === 202,
        },
      );

      if (response.status === 200) {
        setData(response.data as FeatureImportanceData);
        setIsLoading(false);
        stopPolling();
      } else if (response.status === 202) {
        // Still computing — keep polling
        setIsLoading(true);
      }
    } catch (err: any) {
      logger.error("Failed to fetch feature importance:", err);
      setError(err.response?.data?.detail || err.message || "Unknown error");
      setIsLoading(false);
      stopPolling();
    }
  }, [jobId, stopPolling]);

  useEffect(() => {
    if (!jobId) {
      setData(null);
      setIsLoading(false);
      setError(null);
      return;
    }

    // Reset state on jobId change
    setData(null);
    setError(null);
    setIsLoading(true);

    // Fire initial request
    fetchImportance();

    // Start polling
    intervalRef.current = setInterval(fetchImportance, POLL_INTERVAL_MS);

    return () => {
      stopPolling();
    };
  }, [jobId, fetchImportance, stopPolling]);

  return { data, isLoading, error };
}
