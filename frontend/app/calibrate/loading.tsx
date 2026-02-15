import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export default function Loading() {
  return (
    <div className="space-y-6">
      <div>
        <Skeleton className="h-9 w-32" />
        <Skeleton className="mt-2 h-5 w-64" />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-32" />
            <Skeleton className="mt-2 h-4 w-48" />
          </CardHeader>
          <CardContent className="space-y-4">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </CardContent>
        </Card>

        <div className="lg:col-span-2">
          <Card>
            <CardContent className="py-12">
              <div className="flex flex-col items-center justify-center">
                <Skeleton className="h-12 w-12 rounded-full" />
                <Skeleton className="mt-4 h-6 w-32" />
                <Skeleton className="mt-2 h-4 w-48" />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
