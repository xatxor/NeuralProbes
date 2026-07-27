import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "cursor/canvas": path.resolve(root, "src/canvas/index.tsx"),
    },
  },
  server: {
    fs: {
      allow: [root, path.resolve(root, "..")],
    },
  },
});
