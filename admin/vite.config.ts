import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// codec-supervisor mounts the built admin app under `/admin/policies/` —
// the FastAPI app's catch-all proxy forwards anything not under `/admin/*`
// to the inference backend, so this base path keeps the SPA out of the
// proxy lane.
export default defineConfig({
  plugins: [react()],
  base: "/admin/policies/",
  server: {
    port: 5174,
    proxy: {
      // During `npm run dev`, REST calls to /admin/policies/* get forwarded
      // to a locally-running supervisor on :8080.
      "^/admin/policies/(?!index\\.html|assets).*": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
