import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent } from "@/components/ui/card";

export default function ScrapingLoading() {
  return (
    <div className="space-y-6 pb-8">
      <div className="rounded-[28px] border border-border/70 bg-card/90 p-6">
        <div className="space-y-4">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-10 w-3/5" />
          <Skeleton className="h-5 w-4/5" />
          <div className="flex flex-wrap gap-3">
            {[...Array(3)].map((_, index) => (
              <Skeleton key={index} className="h-20 w-44 rounded-2xl" />
            ))}
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[...Array(4)].map((_, index) => (
          <Card key={index} className="border-border/70 bg-card/90">
            <CardContent className="space-y-4 p-5">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-8 w-28" />
              <Skeleton className="h-4 w-36" />
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(380px,0.95fr)]">
        <Card className="border-border/70 bg-card/90">
          <CardContent className="space-y-5 p-6">
            <Skeleton className="h-10 w-64" />
            <Skeleton className="h-16 w-full rounded-2xl" />
            <Skeleton className="h-64 w-full rounded-2xl" />
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card className="border-border/70 bg-card/90">
            <CardContent className="space-y-4 p-6">
              <Skeleton className="h-8 w-48" />
              <Skeleton className="h-20 w-full rounded-2xl" />
              <Skeleton className="h-28 w-full rounded-2xl" />
            </CardContent>
          </Card>
          <Card className="border-border/70 bg-card/90">
            <CardContent className="space-y-3 p-6">
              <Skeleton className="h-8 w-40" />
              {[...Array(4)].map((_, index) => (
                <Skeleton key={index} className="h-24 w-full rounded-2xl" />
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
