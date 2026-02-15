import { Loader2Icon } from "lucide-react";

export default function SchedulesLoading() {
  return (
    <div className="container mx-auto flex min-h-[50vh] items-center justify-center py-10">
      <div className="flex flex-col items-center gap-4">
        <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-muted-foreground">Loading schedules...</p>
      </div>
    </div>
  );
}
