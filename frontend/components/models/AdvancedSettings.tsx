"use client";

import { useState } from "react";
import { Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { DataRangeSection } from "./DataRangeSection";
import { DataSplitSection } from "./DataSplitSection";
import { TrainingSettingsSection } from "./TrainingSettingsSection";
import { defaultConfig } from "./utils";
import type { AdvancedSettingsProps } from "./types";

export function AdvancedSettings({ config, onConfigChange }: AdvancedSettingsProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleReset = () => {
    Object.entries(defaultConfig).forEach(([key, value]) => {
      onConfigChange(key as keyof typeof defaultConfig, value);
    });
  };

  return (
    <Accordion
      type="single"
      collapsible
      value={showAdvanced ? "advanced" : ""}
      onValueChange={(value) => setShowAdvanced(value === "advanced")}
    >
      <AccordionItem value="advanced" className="rounded-lg border">
        <AccordionTrigger className="px-4 py-3 hover:no-underline">
          <div className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            <span className="text-sm font-medium">Advanced Settings</span>
          </div>
        </AccordionTrigger>
        <AccordionContent className="px-4 pb-4">
          <div className="space-y-6 pt-2">
            <DataRangeSection config={config} onConfigChange={onConfigChange} />
            <DataSplitSection config={config} onConfigChange={onConfigChange} />
            <TrainingSettingsSection config={config} onConfigChange={onConfigChange} />
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="w-full"
              onClick={handleReset}
            >
              Reset to Defaults
            </Button>
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
