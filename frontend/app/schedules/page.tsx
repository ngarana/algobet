"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { AlertCircleIcon, Loader2Icon } from "lucide-react";
import { ScheduleCard, ScheduleForm, ExecutionHistory } from "@/components/schedules";
import {
  getSchedules,
  getTaskTypes,
  createSchedule,
  updateSchedule,
  deleteSchedule,
  runScheduleNow,
  getExecutionHistory,
  type ScheduledTask,
  type TaskExecution,
} from "@/lib/api/schedules";

export default function SchedulesPage() {
  const [schedules, setSchedules] = useState<ScheduledTask[]>([]);
  const [taskTypes, setTaskTypes] = useState<string[]>([]);
  const [executionHistories, setExecutionHistories] = useState<
    Record<number, TaskExecution[]>
  >({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<number | null>(null);

  const loadData = useCallback(async () => {
    try {
      const [schedulesResponse, taskTypesResponse] = await Promise.all([
        getSchedules(),
        getTaskTypes(),
      ]);
      setSchedules(schedulesResponse.schedules);
      setTaskTypes(taskTypesResponse);

      // Load execution history for each schedule
      const histories: Record<number, TaskExecution[]> = {};
      for (const schedule of schedulesResponse.schedules) {
        try {
          const history = await getExecutionHistory(schedule.id, 5);
          histories[schedule.id] = history.executions;
        } catch {
          histories[schedule.id] = [];
        }
      }
      setExecutionHistories(histories);
    } catch (err) {
      console.error("Error loading schedules:", err);
      setError("Failed to load schedules");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCreateSchedule = async (data: {
    name: string;
    task_type: string;
    cron_expression: string;
    description?: string;
  }) => {
    try {
      const newSchedule = await createSchedule({
        ...data,
        is_active: true,
      });
      setSchedules((prev) => [...prev, newSchedule]);
    } catch (err) {
      console.error("Error creating schedule:", err);
      throw err;
    }
  };

  const handleRunNow = async (scheduleId: number) => {
    setActionLoading(scheduleId);
    try {
      await runScheduleNow(scheduleId);
      // Refresh execution history
      const history = await getExecutionHistory(scheduleId, 5);
      setExecutionHistories((prev) => ({
        ...prev,
        [scheduleId]: history.executions,
      }));
    } catch (err) {
      console.error("Error running schedule:", err);
      setError("Failed to run schedule");
    } finally {
      setActionLoading(null);
    }
  };

  const handleToggleActive = async (schedule: ScheduledTask) => {
    setActionLoading(schedule.id);
    try {
      const updated = await updateSchedule(schedule.id, {
        is_active: !schedule.is_active,
      });
      setSchedules((prev) => prev.map((s) => (s.id === schedule.id ? updated : s)));
    } catch (err) {
      console.error("Error toggling schedule:", err);
      setError("Failed to update schedule");
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async (scheduleId: number) => {
    if (!confirm("Are you sure you want to delete this schedule?")) return;

    setActionLoading(scheduleId);
    try {
      await deleteSchedule(scheduleId);
      setSchedules((prev) => prev.filter((s) => s.id !== scheduleId));
    } catch (err) {
      console.error("Error deleting schedule:", err);
      setError("Failed to delete schedule");
    } finally {
      setActionLoading(null);
    }
  };

  const getLastExecution = (scheduleId: number): TaskExecution | undefined => {
    return executionHistories[scheduleId]?.[0];
  };

  if (isLoading) {
    return (
      <div className="container mx-auto flex min-h-[50vh] items-center justify-center py-10">
        <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="container mx-auto py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Scheduled Tasks</h1>
        <p className="mt-2 text-muted-foreground">
          Manage automated scraping and prediction tasks
        </p>
      </div>

      {error && (
        <Card className="mb-6 border-destructive">
          <CardContent className="flex items-center gap-2 p-4 text-destructive">
            <AlertCircleIcon className="h-5 w-5" />
            <p>{error}</p>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Schedule List */}
        <div className="space-y-6 lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Active Schedules</CardTitle>
              <CardDescription>
                {schedules.length} scheduled task{schedules.length !== 1 ? "s" : ""}{" "}
                configured
              </CardDescription>
            </CardHeader>
            <CardContent>
              {schedules.length === 0 ? (
                <div className="py-8 text-center text-muted-foreground">
                  No schedules configured. Create one to get started.
                </div>
              ) : (
                <div className="space-y-4">
                  {schedules.map((schedule) => (
                    <div key={schedule.id} className="space-y-4">
                      <ScheduleCard
                        schedule={schedule}
                        lastExecution={getLastExecution(schedule.id)}
                        onRun={() => handleRunNow(schedule.id)}
                        onToggleActive={() => handleToggleActive(schedule)}
                        onDelete={() => handleDelete(schedule.id)}
                        isLoading={actionLoading === schedule.id}
                      />
                      {executionHistories[schedule.id] &&
                        executionHistories[schedule.id].length > 0 && (
                          <Card className="ml-4">
                            <CardHeader className="py-3">
                              <CardTitle className="text-sm">
                                Recent Executions
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="p-0">
                              <ExecutionHistory
                                executions={executionHistories[schedule.id]}
                              />
                            </CardContent>
                          </Card>
                        )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Create Schedule Form */}
        <div className="lg:col-span-1">
          <ScheduleForm taskTypes={taskTypes} onSubmit={handleCreateSchedule} />
        </div>
      </div>
    </div>
  );
}
