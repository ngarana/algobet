"use client";

import type { ReactNode } from "react";

interface FetchFormSectionProps {
  children: ReactNode;
}

export function FetchFormSection({ children }: FetchFormSectionProps) {
  return <div className="space-y-4">{children}</div>;
}
