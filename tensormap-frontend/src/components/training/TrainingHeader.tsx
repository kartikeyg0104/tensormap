/**
 * Training Header Component
 * Displays training status, progress, elapsed time, ETA, and stop button.
 * @module
 */

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { TrainingHeaderProps } from '@/types/training';

const STATUS_COLORS = {
  idle: 'bg-gray-500',
  pending: 'bg-yellow-500',
  running: 'bg-yellow-500',
  completed: 'bg-green-500',
  failed: 'bg-red-500',
  cancelled: 'bg-gray-500',
};

const STATUS_LABELS = {
  idle: 'Idle',
  pending: 'Pending',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

function formatTime(seconds: number): string {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hrs > 0) {
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

export default function TrainingHeader({
  jobId,
  status,
  currentEpoch,
  totalEpochs,
  elapsedTime,
  eta,
  onStop,
  cancelRequested,
}: TrainingHeaderProps) {
  const [displayElapsed, setDisplayElapsed] = useState(elapsedTime);

  // Update elapsed time every second when training is running
  useEffect(() => {
    if (status !== 'running') {
      setDisplayElapsed(elapsedTime);
      return;
    }

    setDisplayElapsed(elapsedTime);
    const interval = setInterval(() => {
      setDisplayElapsed((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [status, elapsedTime]);

  const progress = totalEpochs > 0 ? (currentEpoch / totalEpochs) * 100 : 0;

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex flex-wrap items-center gap-4">
          {/* Status Badge */}
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${STATUS_COLORS[status]}`} />
            <span className="font-medium">{STATUS_LABELS[status]}</span>
          </div>

          {/* Job ID */}
          <div className="text-sm text-gray-600">
            Job: <span className="font-mono">{jobId.slice(0, 8)}</span>
          </div>

          {/* Epoch Progress */}
          <div className="flex-1 min-w-[200px]">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium">
                Epoch {currentEpoch} / {totalEpochs}
              </span>
              <span className="text-sm text-gray-600">{progress.toFixed(0)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>

          {/* Elapsed Time */}
          {status === 'running' && (
            <div className="text-sm">
              <span className="text-gray-600">Elapsed:</span>{' '}
              <span className="font-mono font-medium">{formatTime(displayElapsed)}</span>
            </div>
          )}

          {/* ETA */}
          {status === 'running' && eta > 0 && (
            <div className="text-sm">
              <span className="text-gray-600">ETA:</span>{' '}
              <span className="font-mono font-medium">{formatTime(eta)}</span>
            </div>
          )}

          {/* Stop Button */}
          {status === 'running' && (
            <Button
              variant="destructive"
              size="sm"
              onClick={onStop}
              disabled={cancelRequested}
            >
              {cancelRequested ? 'Stopping...' : 'Stop Training'}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
