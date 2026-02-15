import { FiltersSkeleton, MatchListSkeleton } from "@/components/skeletons";

export default function MatchesLoading() {
  return (
    <div className="space-y-6">
      <div>
        <div className="mb-2 h-8 w-32 animate-pulse rounded bg-muted" />
        <div className="h-4 w-64 animate-pulse rounded bg-muted" />
      </div>

      <FiltersSkeleton />
      <MatchListSkeleton />
    </div>
  );
}
