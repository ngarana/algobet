import { FiltersSkeleton, MatchListSkeleton } from '@/components/skeletons'

export default function MatchesLoading() {
  return (
    <div className="space-y-6">
      <div>
        <div className="h-8 w-32 bg-muted rounded animate-pulse mb-2" />
        <div className="h-4 w-64 bg-muted rounded animate-pulse" />
      </div>

      <FiltersSkeleton />
      <MatchListSkeleton />
    </div>
  )
}
