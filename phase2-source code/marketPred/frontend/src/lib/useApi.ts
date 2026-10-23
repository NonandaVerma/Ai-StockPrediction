import { useEffect, useState } from "react";
import { ApiError } from "../api/client";

export interface AsyncState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
}

/** Run an async loader whenever `deps` change; ignores stale responses. Pass null to skip. */
export function useApi<T>(loader: (() => Promise<T>) | null, deps: unknown[]): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ data: null, error: null, loading: !!loader });
  useEffect(() => {
    if (!loader) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    loader()
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e: unknown) =>
        alive && setState({ data: null, loading: false, error: e instanceof ApiError ? e : new ApiError(-1, String(e)) }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
