import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build straight into the Python package so the built dashboard ships in the
// wheel and `pip install "distill-anything[ui]"` needs no Node at all.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../distillanything/ui/static",
    emptyOutDir: true,
    assetsDir: "assets",
  },
  server: {
    port: 5173,
    proxy: {
      // `npm run dev` against a running `distill ui --no-browser` backend.
      "/api": "http://127.0.0.1:7326",
    },
  },
});
