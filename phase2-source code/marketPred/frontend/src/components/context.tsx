import { createContext, ReactNode, useContext, useEffect, useState } from "react";
import { api } from "../api/client";
import type { Market } from "../api/types";
import { useApi } from "../lib/useApi";

type Theme = "light" | "dark";

interface AppCtx {
  markets: Market[];
  market: Market | null;
  setMarketId: (id: string) => void;
  theme: Theme;
  toggleTheme: () => void;
  loading: boolean;
  error: string | null;
}

const Ctx = createContext<AppCtx | null>(null);

function readStored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key: string, v: string): void {
  try {
    localStorage.setItem(key, v);
  } catch { /* storage unavailable */ }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const { data, error, loading } = useApi(() => api.markets(), []);
  const [marketId, setMarketId] = useState<string>(readStored("mp.market") ?? "SPX");
  const prefersDark = typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  const [theme, setTheme] = useState<Theme>((readStored("mp.theme") as Theme) ?? (prefersDark ? "dark" : "light"));

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    store("mp.theme", theme);
  }, [theme]);
  useEffect(() => store("mp.market", marketId), [marketId]);

  const markets = data ?? [];
  const market = markets.find((m) => m.id === marketId) ?? markets[0] ?? null;
  return (
    <Ctx.Provider value={{
      markets, market, setMarketId, theme, loading,
      toggleTheme: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
      error: error?.message ?? null,
    }}>
      {children}
    </Ctx.Provider>
  );
}

export function useApp(): AppCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useApp outside AppProvider");
  return c;
}
