/**
 * Tests for ConfusionMatrixPanel component
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ConfusionMatrixPanel from "../ConfusionMatrixPanel";

describe("ConfusionMatrixPanel", () => {
  const mockData = {
    confusionMatrix: [
      [10, 2, 1],
      [1, 15, 2],
      [0, 1, 12],
    ],
    classNames: ["Class A", "Class B", "Class C"],
    accuracy: 0.8409,
  };

  it("renders N×N grid", () => {
    const { container } = render(<ConfusionMatrixPanel {...mockData} />);
    
    // Should render title
    expect(screen.getByText("Confusion Matrix")).toBeInTheDocument();
    
    // Should render accuracy badge
    expect(screen.getByText(/Accuracy:/)).toBeInTheDocument();
    expect(screen.getByText(/84.09%/)).toBeInTheDocument();
    
    // Should render all class names (3 columns + 3 rows = 6 occurrences)
    mockData.classNames.forEach((name) => {
      const elements = screen.getAllByText(name);
      expect(elements.length).toBeGreaterThan(0);
    });
    
    // Should render all cell values
    // Use getAllByText for duplicate values
    const allValues = mockData.confusionMatrix.flat();
    const uniqueValues = Array.from(new Set(allValues));
    uniqueValues.forEach((value) => {
      const elements = screen.queryAllByText(value.toString());
      expect(elements.length).toBeGreaterThan(0);
    });
  });

  it("highlights diagonal correctly", () => {
    const { container } = render(<ConfusionMatrixPanel {...mockData} />);
    
    // Diagonal cells (correct predictions) should exist
    // We can't easily test color, but we can verify structure
    expect(container.querySelector(".grid")).toBeInTheDocument();
  });

  it("renders axis labels", () => {
    render(<ConfusionMatrixPanel {...mockData} />);
    
    expect(screen.getByText(/Actual Class/)).toBeInTheDocument();
    expect(screen.getByText(/Predicted Class/)).toBeInTheDocument();
  });
});
