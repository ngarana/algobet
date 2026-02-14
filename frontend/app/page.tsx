'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Calendar, Trophy, TrendingUp, Activity } from 'lucide-react'
import { useUpcomingPredictions } from '@/lib/queries/use-predictions'
import { useActiveModel } from '@/lib/queries/use-models'
import { useValueBets } from '@/lib/queries/use-value-bets'

function UpcomingMatchesCard() {
  const { data, isLoading } = useUpcomingPredictions(7)
  const count = data?.items?.length ?? 0

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">Upcoming Matches</CardTitle>
        <Calendar className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-8 w-20" />
        ) : (
          <div className="text-2xl font-bold">{count}</div>
        )}
        <p className="text-xs text-muted-foreground">Next 7 days</p>
      </CardContent>
    </Card>
  )
}

function ActiveModelCard() {
  const { data, isLoading } = useActiveModel()

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">Active Model</CardTitle>
        <Trophy className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-8 w-32" />
        ) : data ? (
          <>
            <div className="text-2xl font-bold">{data.name}</div>
            <p className="text-xs text-muted-foreground">
              {data.algorithm} • {data.accuracy ? `${(data.accuracy * 100).toFixed(1)}%` : 'N/A'} accuracy
            </p>
          </>
        ) : (
          <>
            <div className="text-2xl font-bold">No Active Model</div>
            <p className="text-xs text-muted-foreground">Activate a model to start predictions</p>
          </>
        )}
      </CardContent>
    </Card>
  )
}

function ValueBetsCard() {
  const { data, isLoading } = useValueBets({ limit: 10 })
  const count = data?.items?.length ?? 0

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">Value Bets</CardTitle>
        <TrendingUp className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-8 w-20" />
        ) : (
          <div className="text-2xl font-bold">{count}</div>
        )}
        <p className="text-xs text-muted-foreground">Opportunities available</p>
      </CardContent>
    </Card>
  )
}

function PredictionAccuracyCard() {
  const { data, isLoading } = useUpcomingPredictions()
  
  // Calculate average confidence
  const avgConfidence = data?.items?.length
    ? data.items.reduce((sum, p) => sum + p.confidence, 0) / data.items.length
    : 0

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
        <Activity className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-8 w-20" />
        ) : (
          <div className="text-2xl font-bold">
            {avgConfidence ? `${(avgConfidence * 100).toFixed(1)}%` : 'N/A'}
          </div>
        )}
        <p className="text-xs text-muted-foreground">Across predictions</p>
      </CardContent>
    </Card>
  )
}

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Overview of your football predictions and analysis
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <UpcomingMatchesCard />
        <ActiveModelCard />
        <ValueBetsCard />
        <PredictionAccuracyCard />
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Dashboard components will be populated with data from the API.
              Ensure the backend is running and has data.
            </p>
          </CardContent>
        </Card>

        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>Quick Stats</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Quick statistics and insights will appear here once data is available.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
