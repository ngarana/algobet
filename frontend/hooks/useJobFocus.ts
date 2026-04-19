/**
 * Hook for managing focused job state and WebSocket subscription logic
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import type { FetchJob } from "@/lib/api/fetch";
import { ACTIVE_JOB_STATUSES } from "@/lib/constants/fetch";

/**
 * Check if a job is currently active (pending or running)
 */
export function isActiveJob(job: FetchJob): boolean {
  return ACTIVE_JOB_STATUSES.includes(
    job.status as (typeof ACTIVE_JOB_STATUSES)[number]
  );
}

interface UseJobFocusOptions {
  jobs: FetchJob[];
  autoSelectFirst?: boolean;
}

interface UseJobFocusReturn {
  focusedJobId: string | null;
  focusedJob: FetchJob | null;
  activeJobs: FetchJob[];
  liveSocketJobId: string | undefined;
  setFocusedJobId: (id: string | null) => void;
}

/**
 * Hook to manage job focus state and determine which job to track via WebSocket
 */
export function useJobFocus(options: UseJobFocusOptions): UseJobFocusReturn {
  const { jobs, autoSelectFirst = true } = options;
  const [focusedJobId, setFocusedJobId] = useState<string | null>(null);

  // Derive active jobs
  const activeJobs = useMemo(() => jobs.filter(isActiveJob), [jobs]);

  // Find focused job
  const focusedJob = useMemo(
    () => jobs.find((job) => job.id === focusedJobId) ?? null,
    [jobs, focusedJobId]
  );

  // Determine which job ID to use for WebSocket subscription
  // Priority: focused active job > first active job > undefined
  const liveSocketJobId = useMemo(() => {
    if (focusedJob && isActiveJob(focusedJob)) return focusedJob.id;
    if (activeJobs.length > 0) return activeJobs[0].id;
    return undefined;
  }, [focusedJob, activeJobs]);

  // Auto-select job when jobs change
  useEffect(() => {
    if (jobs.length === 0) {
      if (focusedJobId !== null) setFocusedJobId(null);
      return;
    }

    const focusedStillExists = jobs.some((job) => job.id === focusedJobId);
    if (!focusedStillExists && autoSelectFirst) {
      setFocusedJobId(activeJobs[0]?.id ?? jobs[0]?.id ?? null);
    }
  }, [activeJobs, focusedJobId, jobs, autoSelectFirst]);

  return {
    focusedJobId,
    focusedJob,
    activeJobs,
    liveSocketJobId,
    setFocusedJobId,
  };
}
