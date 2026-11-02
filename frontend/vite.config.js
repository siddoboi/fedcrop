import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the page calls /api/... on its own origin and Vite forwards
// those requests to the FastAPI service, so no CORS setup is needed.
// 127.0.0.1, not localhost: on Windows, localhost tries IPv6 first and every
// request waits about 2 seconds before falling back.
const API = process.env.VITE_API_PROXY || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 800 }, // Recharts alone is ~400 kB; one bundle is fine for a local dashboard
  server: { port: 5173, proxy: { "/api": { target: API, changeOrigin: true } } },
  preview: { port: 4173, proxy: { "/api": { target: API, changeOrigin: true } } },
});
