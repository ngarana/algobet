"use client";

import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CheckCircleIcon, XCircleIcon, ClockIcon, Loader2Icon } from "lucide-react";
import type { TaskExecution } from "@/lib/api/schedules";

interface ExecutionHistoryProps {
  executions: TaskExecution[];
  isLoading?: boolean;
}

function getStatusIcon(status: string) {
  switch (status) {
    case "completed":
      return <CheckCircleIcon className="h-4 w-4 text-green-500" />;
    case "failed":
      return <XCircleIcon className="h-4 w-4 text-destructive" />;
    case "running":
      return <Loader2Icon className="h-4 w-4 animate-spin text-primary" />;
    default:
      return <ClockIcon className="h-4 w-4 text-muted-foreground" />;
  }
}

function getStatusBadge(status: string) {
  switch (status) {
    case "completed":
      return <Badge variant="success">Completed</Badge>;
    case "failed":
      return <Badge variant="destructive">Failed</Badge>;
    case "running":
      return <Badge variant="default">Running</Badge>;
    default:
      return <Badge variant="secondary">{status}</Badge>;
  }
}

export function ExecutionHistory({
  executions,
  isLoading = false,
}: ExecutionHistoryProps) {
  if (isLoading) {
    return (
      <div className="flex h-32 items-center justify-center">
        <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (executions.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center text-muted-foreground">
        No execution history
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-12">Status</TableHead>
          <TableHead>Started</TableHead>
          <TableHead>Completed</TableHead>
          <TableHead>Duration</TableHead>
          <TableHead>Result</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {executions.map((execution) => (
          <TableRow key={execution.id}>
            <TableCell>
              <div className="flex items-center gap-2">
                {getStatusIcon(execution.status)}
                {getStatusBadge(execution.status)}
              </div>
            </TableCell>
            <TableCell className="text-sm">
              {new Date(execution.started_at).toLocaleString()}
            </TableCell>
            <TableCell className="text-sm">
              {execution.completed_at
                ? new Date(execution.completed_at).toLocaleString()
                : "-"}
            </TableCell>
            <TableCell className="text-sm">
              {execution.duration !== null && execution.duration !== undefined
                ? `${execution.duration.toFixed(2)}s`
                : "-"}
            </TableCell>
            <TableCell className="text-sm">
              {execution.error_message ? (
                <span className="text-destructive">{execution.error_message}</span>
              ) : execution.result ? (
                <span className="text-muted-foreground">
                  {typeof execution.result.status === "string"
                    ? execution.result.status
                    : "OK"}
                </span>
              ) : (
                "-"
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
