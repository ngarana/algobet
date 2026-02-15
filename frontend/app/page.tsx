'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { 
  Calendar, 
  Trophy, 
  TrendingUp, 
  Activity, 
  DollarSign, 
  Target, 
  BarChart3,
  RefreshCw
} from 'lucide-react'
import { useUpcomingPredictions } from '@/lib/queries/use-predictions'
import { useActiveModel } from '@/lib/queries/use-models'
import { useValueBets } from '@/lib/queries/use-value-bets'
import { useDashboardStats } from '@/lib/queries/use-dashboard-stats'
import { PerformanceTrendsChart } from '@/components/charts/performance-trends-chart'
import { RecentActivityFeed } from '@/components/dashboard/recent-activity-feed'
import { QuickActionsPanel } from '@/components/dashboard/quick-actions-panel'

import Link from 'next/link'

// Updated Upcoming Matches Card to show all upcoming matches
function UpcomingMatchesCard() {
  const { data, isLoading } = useUpcomingPredictions() // Removed days parameter to get all upcoming matches
  const count = data?.items?.length ?? 0

  return (
    <Link href="/matches" className="block">
      <Card className="hover:shadow-md transition-shadow cursor-pointer">
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
          <p className="text-xs text-muted-foreground">All upcoming matches</p>
        </CardContent>
      </Card>
    </Link>
  )
}

function ActiveModelCard() {
  const { data, isLoading } = useActiveModel()

  return (
    <Link href="/models" className="block">
      <Card className="hover:shadow-md transition-shadow cursor-pointer">
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
    </Link>
  )
}

function ValueBetsCard() {
  const { data, isLoading } = useValueBets({ max_matches: 100 })
  const count = data?.length ?? 0

  return (
    <Link href="/value-bets" className="block">
      <Card className="hover:shadow-md transition-shadow cursor-pointer">
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
    </Link>
  )
}

function PredictionAccuracyCard() {
  const { data, isLoading } = useUpcomingPredictions()

  // Calculate average confidence
  const avgConfidence = data?.items?.length
    ? data.items.reduce((sum, p) => sum + p.confidence, 0) / data.items.length
    : 0

  return (
    <Link href="/predictions" className="block">
      <Card className="hover:shadow-md transition-shadow cursor-pointer">
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
    </Link>
  )
}

// New Profit/Loss Card
function ProfitLossCard() {
  const { data, isLoading } = useDashboardStats()
  
  return (
    <Link href="/predictions" className="block">
      <Card className="hover:shadow-md transition-shadow cursor-pointer">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Profit/Loss</CardTitle>
          <DollarSign className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-8 w-20" />
          ) : (
            <div className={`text-2xl font-bold ${data?.totalProfit && data.totalProfit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              ${data?.totalProfit?.toFixed(2) || '0.00'}
            </div>
          )}
          <p className="text-xs text-muted-foreground">Total since inception</p>
        </CardContent>
      </Card>
    </Link>
  )
}

// New Win Rate Card
function WinRateCard() {
  const { data, isLoading } = useDashboardStats()
  
  return (
    <Link href="/predictions" className="block">
      <Card className="hover:shadow-md transition-shadow cursor-pointer">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
          <Target className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-8 w-20" />
          ) : (
            <div className="text-2xl font-bold">
              {data?.winRate ? `${(data.winRate * 100).toFixed(1)}%` : '0%'}
            </div>
          )}
          <p className="text-xs text-muted-foreground">Successful predictions</p>
        </CardContent>
      </Card>
    </Link>
  )
}

function PerformanceTrendsCard() {
  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Performance Trends
        </CardTitle>
      </CardHeader>
      <CardContent>
        <PerformanceTrendsChart />
      </CardContent>
    </Card>
  )
}

function QuickActionsCard() {
  return (
    <Card className="col-span-3">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Quick Actions
        </CardTitle>
      </CardHeader>
      <CardContent>
        <QuickActionsPanel />
      </CardContent>
    </Card>
  )
}

function RecentActivityCard() {
  return (
    <Card className="col-span-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Recent Activity
        </CardTitle>
      </CardHeader>
      <CardContent>
        <RecentActivityFeed />
      </CardContent>
    </Card>
  )
}

export default function DashboardPage() {
  const [refreshKey, setRefreshKey] = useState(0)
  
  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1)
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Overview of your football predictions and analysis
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={handleRefresh}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Top Row - Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-6">
        <UpcomingMatchesCard key={`upcoming-${refreshKey}`} />
        <ActiveModelCard key={`active-model-${refreshKey}`} />
        <ValueBetsCard key={`value-bets-${refreshKey}`} />
        <PredictionAccuracyCard key={`accuracy-${refreshKey}`} />
        <ProfitLossCard key={`profit-loss-${refreshKey}`} />
        <WinRateCard key={`win-rate-${refreshKey}`} />
      </div>

      {/* Charts and Quick Actions Row */}
      <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-7">
        <PerformanceTrendsCard />
        <QuickActionsCard />
      </div>

      {/* Recent Activity */}
      <RecentActivityCard />
    </div>
  )
}