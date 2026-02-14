import { Loader2Icon } from 'lucide-react'

export default function SchedulesLoading() {
  return (
    <div className="container mx-auto py-10 flex justify-center items-center min-h-[50vh]">
      <div className="flex flex-col items-center gap-4">
        <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-muted-foreground">Loading schedules...</p>
      </div>
    </div>
  )
}
