"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity } from "lucide-react";

interface ActivityItem {
  id: string;
  type: "prediction" | "value_bet" | "model_update" | "profit";
  title: string;
  description: string;
  timestamp: string;
  color: string;
}

interface RecentActivityFeedProps {
  activities?: ActivityItem[];
  isLoading?: boolean;
}

export function RecentActivityFeed({
  activities: propActivities,
  isLoading = false,
}: RecentActivityFeedProps) {
  const [activities, setActivities] = useState<ActivityItem[]>(
    propActivities || [
      {
        id: "1",
        type: "value_bet",
        title: "New Value Bet Found",
        description: "Manchester United vs Liverpool @ 2.45 odds",
        timestamp: "2 hours ago",
        color: "bg-green-500",
      },
      {
        id: "2",
        type: "model_update",
        title: "Model Updated",
        description: "Random Forest v2.1 activated",
        timestamp: "5 hours ago",
        color: "bg-blue-500",
      },
      {
        id: "3",
        type: "prediction",
        title: "Prediction Generated",
        description: "15 new match predictions",
        timestamp: "Yesterday",
        color: "bg-purple-500",
      },
      {
        id: "4",
        type: "profit",
        title: "Profit Recorded",
        description: "+$245.50 from recent bets",
        timestamp: "2 days ago",
        color: "bg-yellow-500",
      },
    ]
  );

  // In a real implementation, this would fetch live data
  useEffect(() => {
    // Simulate fetching recent activities
  }, []);

  if (isLoading) {
    return (
      <div className="space-y-3">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="h-2 w-2 animate-pulse rounded-full bg-gray-300"></div>
            <div className="flex-1 space-y-2">
              <div className="h-4 w-3/4 animate-pulse rounded bg-gray-200"></div>
              <div className="h-3 w-1/2 animate-pulse rounded bg-gray-200"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {activities.map((activity) => (
        <div key={activity.id} className="flex items-start gap-3">
          <div className={`mt-1.5 h-2 w-2 rounded-full ${activity.color}`}></div>
          <div className="min-w-0 flex-1">
            <p className="text-sm">
              <span className="font-medium">{activity.title}:</span>{" "}
              {activity.description}
            </p>
            <p className="text-xs text-muted-foreground">{activity.timestamp}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
