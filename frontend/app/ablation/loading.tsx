export default function LoadingAblation() {
  return (
    <div className="space-y-6">
      <div>
        <div className="h-9 w-64 animate-pulse rounded bg-muted" />
        <div className="mt-2 h-5 w-96 animate-pulse rounded bg-muted" />
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6">
          <div className="h-[500px] animate-pulse rounded-lg bg-muted" />
        </div>
        <div className="space-y-6 lg:col-span-2">
          <div className="h-[300px] animate-pulse rounded-lg bg-muted" />
        </div>
      </div>
    </div>
  );
}
