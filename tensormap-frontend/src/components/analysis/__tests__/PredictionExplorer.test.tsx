/**
 * Tests for PredictionExplorer component
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import PredictionExplorer from "../PredictionExplorer";
import axios from "@/shared/Axios";

vi.mock("@/shared/Axios");

describe("PredictionExplorer", () => {
  const mockResponse = {
    data: {
      total: 100,
      offset: 0,
      limit: 25,
      predictions: [
        {
          index: 0,
          actual_class: 0,
          actual_class_name: "Class A",
          predicted_class: 0,
          predicted_class_name: "Class A",
          confidence: 0.95,
          probabilities: [0.95, 0.03, 0.02],
          features: { f1: 1.2, f2: 3.4 },
          is_correct: true,
        },
        {
          index: 1,
          actual_class: 1,
          actual_class_name: "Class B",
          predicted_class: 2,
          predicted_class_name: "Class C",
          confidence: 0.65,
          probabilities: [0.15, 0.20, 0.65],
          features: { f1: 2.1, f2: 4.5 },
          is_correct: false,
        },
      ],
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows correct/incorrect badges", async () => {
    (axios.get as any).mockResolvedValue(mockResponse);

    render(<PredictionExplorer jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText("Correct")).toBeInTheDocument();
      expect(screen.getByText("Misclassified")).toBeInTheDocument();
    });
  });

  it("filter works", async () => {
    (axios.get as any).mockResolvedValue(mockResponse);

    const { container } = render(<PredictionExplorer jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText("Correct")).toBeInTheDocument();
    });

    // Filter dropdown should exist
    const selectTrigger = container.querySelector('[role="combobox"]');
    expect(selectTrigger || screen.getByText(/All Predictions/i)).toBeInTheDocument();
  });

  it("pagination controls render", async () => {
    (axios.get as any).mockResolvedValue(mockResponse);

    render(<PredictionExplorer jobId="test-job-123" />);

    await waitFor(() => {
      // Should show pagination info
      expect(screen.getByText(/Showing.*of.*100/)).toBeInTheDocument();
    });
  });

  it("handles loading state", () => {
    (axios.get as any).mockImplementation(
      () => new Promise(() => {}), // Never resolves
    );

    render(<PredictionExplorer jobId="test-job-123" />);

    // Should show loading spinner
    const spinner = document.querySelector(".animate-spin");
    expect(spinner).toBeInTheDocument();
  });

  it("handles error state", async () => {
    (axios.get as any).mockRejectedValue({
      response: { data: { detail: "Test error" } },
    });

    render(<PredictionExplorer jobId="test-job-123" />);

    await waitFor(() => {
      expect(screen.getByText(/Test error/)).toBeInTheDocument();
    });
  });
});
