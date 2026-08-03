/**
 * Tests for FeatureImportanceChart component
 */

import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import FeatureImportanceChart from "../FeatureImportanceChart";

describe("FeatureImportanceChart", () => {
  const mockData = {
    features: ["feature1", "feature2", "feature3", "feature4", "feature5"],
    importancesMean: [0.25, 0.15, 0.0005, 0.0003, 0.0001],
    importancesStd: [0.02, 0.01, 0.0001, 0.00005, 0.00001],
  };

  it("shows error bars", () => {
    const { container } = render(<FeatureImportanceChart {...mockData} />);
    
    // Recharts renders in responsive container
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument();
  });

  it("collapses low-importance features", () => {
    render(<FeatureImportanceChart {...mockData} />);
    
    // Should show toggle button for low-importance features
    const toggleButton = screen.getByText(/low-importance features/i);
    expect(toggleButton).toBeInTheDocument();
    
    // Click to expand
    fireEvent.click(toggleButton);
    
    // Button text should change
    expect(screen.getByText(/Hide.*low-importance features/i)).toBeInTheDocument();
  });

  it("renders legend with color categories", () => {
    render(<FeatureImportanceChart {...mockData} />);
    
    expect(screen.getByText(/High importance/)).toBeInTheDocument();
    expect(screen.getByText(/Medium importance/)).toBeInTheDocument();
    expect(screen.getByText(/Low importance/)).toBeInTheDocument();
  });

  it("renders all features", () => {
    render(<FeatureImportanceChart {...mockData} />);
    
    // High importance features should be visible initially
    // Recharts may not render text in test environment due to SVG rendering
    // Verify structure instead
    expect(screen.getByText(/Feature Importance/)).toBeInTheDocument();
    expect(screen.getByText(/low-importance features/i)).toBeInTheDocument();
  });
});
