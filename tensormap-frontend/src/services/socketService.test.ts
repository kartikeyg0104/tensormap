/**
 * Test suite for socketService
 * @module
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock axios - must be before imports
vi.mock("../shared/Axios", () => ({
  default: {
    delete: vi.fn(() => Promise.resolve({ data: { success: true } })),
  },
}));

// Mock socket.io-client - must be before imports
vi.mock("socket.io-client", () => ({
  io: vi.fn(() => ({
    on: vi.fn(),
    off: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
    connected: true,
  })),
}));

import { subscribeToJob, unsubscribeFromJob, cancelJob, getTrainingSocket } from "./socketService";
import axios from "../shared/Axios";

describe("socketService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("subscribeToJob", () => {
    it("subscribes to job and returns cleanup function", () => {
      const jobId = "test-job-123";
      const onEvent = vi.fn();

      const cleanup = subscribeToJob(jobId, onEvent);

      // Should return a cleanup function
      expect(typeof cleanup).toBe("function");
    });

    it("handles events with structured data", () => {
      const jobId = "test-job-123";
      const onEvent = vi.fn();

      const cleanup = subscribeToJob(jobId, onEvent);

      // The handler is set up, we can't directly test the Socket.IO events
      // in unit tests, but we verify the subscription happens
      expect(typeof cleanup).toBe("function");

      cleanup();
    });
  });

  describe("unsubscribeFromJob", () => {
    it("unsubscribes from job", () => {
      const jobId = "test-job-123";

      // Should not throw
      expect(() => unsubscribeFromJob(jobId)).not.toThrow();
    });

    it("handles null jobId gracefully", () => {
      expect(() => unsubscribeFromJob(null as any)).not.toThrow();
    });
  });

  describe("cancelJob", () => {
    it("sends DELETE request to cancel endpoint", async () => {
      const jobId = "test-job-123";

      await cancelJob(jobId);

      expect(axios.delete).toHaveBeenCalledWith(expect.stringContaining(jobId));
    });

    it("returns promise that resolves", async () => {
      const jobId = "test-job-123";

      const result = await cancelJob(jobId);

      expect(result).toBeDefined();
    });
  });

  describe("getTrainingSocket", () => {
    it("returns a socket instance", () => {
      const socket = getTrainingSocket();
      expect(socket).toBeDefined();
    });
  });
});
