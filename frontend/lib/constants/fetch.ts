/**
 * Constants for fetch operations
 */

/** Dialog types for fetch operations */
export const FetchDialogType = {
  UPCOMING: "upcoming",
  RESULTS: "results",
  BY_DATE: "by-date",
} as const;

export type FetchDialogTypeValue =
  (typeof FetchDialogType)[keyof typeof FetchDialogType];

/** Log levels for live stream */
export const LogLevel = {
  INFO: "INF",
  SUCCESS: "SUC",
  SYSTEM: "SYS",
  ERROR: "ERR",
} as const;

export type LogLevelValue = (typeof LogLevel)[keyof typeof LogLevel];

/** Job status constants */
export const JobStatus = {
  PENDING: "pending",
  RUNNING: "running",
  COMPLETED: "completed",
  FAILED: "failed",
  CANCELLED: "cancelled",
} as const;

export type JobStatusValue = (typeof JobStatus)[keyof typeof JobStatus];

/** Active job statuses */
export const ACTIVE_JOB_STATUSES: JobStatusValue[] = [
  JobStatus.PENDING,
  JobStatus.RUNNING,
];

/** Terminal job statuses (job is finished) */
export const TERMINAL_JOB_STATUSES: JobStatusValue[] = [
  JobStatus.COMPLETED,
  JobStatus.FAILED,
  JobStatus.CANCELLED,
];

/** Dialog configuration */
export const FETCH_DIALOG_CONFIG = {
  [FetchDialogType.UPCOMING]: {
    title: "New Upcoming Fetch Job",
    endpoint: "SCRAPE /upcoming",
    color: "#4ade80",
  },
  [FetchDialogType.RESULTS]: {
    title: "New Results Fetch Job",
    endpoint: "SCRAPE /results",
    color: "#f59e0b",
  },
  [FetchDialogType.BY_DATE]: {
    title: "New Date Fetch Job",
    endpoint: "SCRAPE /by-date",
    color: "#8b5cf6",
  },
} as const;

/** Error messages */
export const ERROR_MESSAGES = {
  FETCH_FAILED: "Failed to start fetch operation. Check that the scraper is running.",
  TOURNAMENT_URL_REQUIRED: "Select a league before starting historical scraping",
} as const;
