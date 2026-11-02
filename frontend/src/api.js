import { useEffect, useState } from "react";

// Empty base means "same origin": in development Vite proxies /api to FastAPI.
// Set VITE_API_BASE only when the built site is served apart from the API.
const BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

// The results are frozen, so a response never changes while the page is open.
// Caching per URL means switching views does not refetch.
const cache = new Map();

export function apiUrl(path) {
  return BASE + path;
}

export async function getJson(path) {
  if (cache.has(path)) return cache.get(path);
  const p = fetch(apiUrl(path)).then(async (res) => {
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
      throw new Error(`${res.status} ${detail}`);
    }
    return res.json();
  });
  cache.set(path, p);
  p.catch(() => cache.delete(path)); // let a failed request be retried
  return p;
}

/** Fetch one or more API paths. Returns { data, error, loading }.
 *  data is a single object for one path, an array for several. */
export function useApi(...paths) {
  const key = paths.join("|");
  const [state, setState] = useState({ data: null, error: null, loading: true });

  useEffect(() => {
    let live = true;
    setState({ data: null, error: null, loading: true });
    Promise.all(paths.map(getJson))
      .then((res) => live && setState({
        data: paths.length === 1 ? res[0] : res, error: null, loading: false }))
      .catch((err) => live && setState({ data: null, error: err, loading: false }));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return state;
}
