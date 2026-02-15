'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { BarChart3 } from 'lucide-react'

interface PerformanceTrendsProps {
  data?: any[]
  isLoading?: boolean
}

export function PerformanceTrendsChart({ data, isLoading = false }: PerformanceTrendsProps) {
  if (isLoading) {
    return (
      <div className="h-64">
        <Skeleton className="h-full w-full" />
      </div>
    )
  }

  return (
    <div className="h-64 flex flex-col">
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        <div className="text-center">
          <BarChart3 className="h-12 w-12 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Performance chart will be displayed here</p>
          <p className="text-xs mt-1 text-muted-foreground">
            This chart will show your ROI and profit trends over time
          </p>
        </div>
      </div>
      <div className="flex justify-center gap-4 mt-4">
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-green-500"></div>
          <span className="text-xs">Profit</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-blue-500"></div>
          <span className="text-xs">ROI</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-purple-500"></div>
          <span className="text-xs">Predictions</span>
        </div>
      </div>
    </div>
  )
}