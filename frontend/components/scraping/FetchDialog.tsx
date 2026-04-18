"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { FetchDialogType, type FetchDialogTypeValue } from "@/lib/constants/fetch";
import { FETCH_DIALOG_CONFIG } from "@/lib/constants/fetch";
import { PlayIcon } from "lucide-react";

interface FetchDialogProps {
  type: FetchDialogTypeValue;
  onConfirm: (data: { tournamentUrl?: string; date?: string }) => void;
  onClose: () => void;
  isLoading?: boolean;
}

/**
 * Unified dialog component for all fetch operation types
 */
export function FetchDialog({
  type,
  onConfirm,
  onClose,
  isLoading = false,
}: FetchDialogProps) {
  const [tournamentUrl, setTournamentUrl] = useState("");
  const [date, setDate] = useState("");

  const config = FETCH_DIALOG_CONFIG[type];

  const handleClose = () => {
    setTournamentUrl("");
    setDate("");
    onClose();
  };

  const handleConfirm = () => {
    if (type === FetchDialogType.UPCOMING) {
      onConfirm({ tournamentUrl: tournamentUrl || undefined });
    } else if (type === FetchDialogType.RESULTS) {
      onConfirm({ tournamentUrl });
    } else if (type === FetchDialogType.BY_DATE) {
      onConfirm({ date: date || undefined });
    }
    handleClose();
  };

  const isConfirmDisabled =
    isLoading || (type === FetchDialogType.RESULTS && !tournamentUrl);

  return (
    <Card className="border-[#252a37] bg-[#12151d]">
      <CardContent className="p-6">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-[#e0e6f0]">{config.title}</h2>
          <button
            onClick={handleClose}
            className="text-[#9ca3af] hover:text-[#e0e6f0]"
            aria-label="Close dialog"
          >
            ✕
          </button>
        </div>

        {type === FetchDialogType.UPCOMING && (
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm text-[#9ca3af]">
                Tournament URL (optional)
              </label>
              <input
                type="text"
                value={tournamentUrl}
                onChange={(e) => setTournamentUrl(e.target.value)}
                placeholder="https://www.oddsportal.com/soccer/england/premier-league/"
                className="w-full rounded-md border border-[#252a37] bg-[#161a25] px-3 py-2 text-sm text-[#e0e6f0] placeholder:text-[#444c5e]"
              />
            </div>
            <DialogActions
              onConfirm={handleConfirm}
              onClose={handleClose}
              isLoading={isLoading}
              confirmDisabled={isConfirmDisabled}
              confirmColor={config.color}
            />
          </div>
        )}

        {type === FetchDialogType.RESULTS && (
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm text-[#9ca3af]">
                Tournament URL <span className="text-[#f87171]">*</span>
              </label>
              <input
                type="text"
                value={tournamentUrl}
                onChange={(e) => setTournamentUrl(e.target.value)}
                placeholder="https://www.oddsportal.com/soccer/england/premier-league/"
                className="w-full rounded-md border border-[#252a37] bg-[#161a25] px-3 py-2 text-sm text-[#e0e6f0] placeholder:text-[#444c5e]"
                required
              />
            </div>
            <DialogActions
              onConfirm={handleConfirm}
              onClose={handleClose}
              isLoading={isLoading}
              confirmDisabled={isConfirmDisabled}
              confirmColor={config.color}
            />
          </div>
        )}

        {type === FetchDialogType.BY_DATE && (
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm text-[#9ca3af]">
                Date (YYYY-MM-DD)
              </label>
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="w-full rounded-md border border-[#252a37] bg-[#161a25] px-3 py-2 text-sm text-[#e0e6f0]"
              />
            </div>
            <DialogActions
              onConfirm={handleConfirm}
              onClose={handleClose}
              isLoading={isLoading}
              confirmDisabled={isConfirmDisabled}
              confirmColor={config.color}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface DialogActionsProps {
  onConfirm: () => void;
  onClose: () => void;
  isLoading: boolean;
  confirmDisabled: boolean;
  confirmColor: string;
}

function DialogActions({
  onConfirm,
  onClose,
  isLoading,
  confirmDisabled,
  confirmColor,
}: DialogActionsProps) {
  return (
    <div className="flex gap-2">
      <Button
        onClick={onConfirm}
        disabled={confirmDisabled}
        className="font-semibold text-[#0a0c12]"
        style={{ backgroundColor: confirmColor }}
      >
        <PlayIcon className="mr-2 h-4 w-4" />
        Start Fetch
      </Button>
      <Button
        variant="outline"
        onClick={onClose}
        className="border-[#252a37] text-[#9ca3af]"
      >
        Cancel
      </Button>
    </div>
  );
}
