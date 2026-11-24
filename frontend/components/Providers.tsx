"use client";
import { SettingsProvider } from "@/lib/settings";
import { ToastProvider } from "@/components/Toast";
import { TooltipProvider } from "@/components/Tooltip";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SettingsProvider>
      <ToastProvider>
        <TooltipProvider>{children}</TooltipProvider>
      </ToastProvider>
    </SettingsProvider>
  );
}
