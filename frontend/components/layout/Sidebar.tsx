"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/stores/ui-store";
import {
  LayoutDashboard,
  Calendar,
  Trophy,
  Users,
  Brain,
  TrendingUp,
  Settings,
} from "lucide-react";

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Matches", href: "/matches", icon: Calendar },
  { name: "Predictions", href: "/predictions", icon: Brain },
  { name: "Value Bets", href: "/value-bets", icon: TrendingUp },
  { name: "Models", href: "/models", icon: Trophy },
  { name: "Teams", href: "/teams", icon: Users },
  { name: "Scraping", href: "/scraping", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const { sidebarOpen } = useUIStore();

  return (
    <aside
      className={cn(
        "fixed left-0 top-14 z-40 h-[calc(100vh-3.5rem)] w-64 border-r bg-background transition-transform duration-300 ease-in-out md:translate-x-0",
        sidebarOpen ? "translate-x-0" : "-translate-x-full md:hidden"
      )}
    >
      <nav className="flex flex-col gap-1 p-4">
        {navigation.map((item) => {
          const isActive =
            pathname === item.href || pathname.startsWith(`${item.href}/`);
          const Icon = item.icon;

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <Icon className="h-5 w-5" />
              {item.name}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
