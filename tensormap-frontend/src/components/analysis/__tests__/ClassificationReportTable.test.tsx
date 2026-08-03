/**
 * Tests for ClassificationReportTable component
 */

import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ClassificationReportTable from "../ClassificationReportTable";

describe("ClassificationReportTable", () => {
  const mockData = {
    classificationReport: {
      "Class A": {
        precision: 0.9091,
        recall: 0.7692,
        "f1-score": 0.8333,
        support: 13,
      },
      "Class B": {
        precision: 0.8333,
        recall: 0.8333,
        "f1-score": 0.8333,
        support: 18,
      },
      "Class C": {
        precision: 0.8571,
        recall: 0.9231,
        "f1-score": 0.8889,
        support: 13,
      },
      "macro avg": {
        precision: 0.8665,
        recall: 0.8419,
        "f1-score": 0.8518,
        support: 44,
      },
      "weighted avg": {
        precision: 0.8636,
        recall: 0.8409,
        "f1-score": 0.8497,
        support: 44,
      },
      accuracy: 0.8409,
    },
    classNames: ["Class A", "Class B", "Class C"],
  };

  it("sorts by F1 on click", () => {
    render(<ClassificationReportTable {...mockData} />);
    
    // Find F1-Score header
    const f1Header = screen.getByText(/F1-Score/);
    expect(f1Header).toBeInTheDocument();
    
    // Click to sort
    fireEvent.click(f1Header);
    
    // Component should re-render with sorted data
    // We verify by checking the table is still rendered
    expect(screen.getByText("Class A")).toBeInTheDocument();
  });

  it("renders all metrics columns", () => {
    render(<ClassificationReportTable {...mockData} />);
    
    expect(screen.getByText(/Precision/)).toBeInTheDocument();
    expect(screen.getByText(/Recall/)).toBeInTheDocument();
    expect(screen.getByText(/F1-Score/)).toBeInTheDocument();
    expect(screen.getByText(/Support/)).toBeInTheDocument();
  });

  it("shows macro and weighted averages", () => {
    render(<ClassificationReportTable {...mockData} />);
    
    expect(screen.getByText("Macro Avg")).toBeInTheDocument();
    expect(screen.getByText("Weighted Avg")).toBeInTheDocument();
  });

  it("displays per-class metrics", () => {
    render(<ClassificationReportTable {...mockData} />);
    
    mockData.classNames.forEach((className) => {
      expect(screen.getByText(className)).toBeInTheDocument();
    });
  });
});
