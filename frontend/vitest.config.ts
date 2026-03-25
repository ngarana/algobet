import { defineConfig } from "vitest/config";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setupTests.ts"],
    include: ["**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}"],
    exclude: ["node_modules", ".next", "out", "playwright-report", "coverage"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
      include: [
        "src/**/*.{ts,tsx}",
        "components/**/*.{ts,tsx}",
        "hooks/**/*.{ts,tsx}",
        "lib/**/*.{ts,tsx}",
      ],
      exclude: [
        "src/**/*.d.ts",
        "src/**/*.{test,spec}.{ts,tsx}",
        "src/test/**",
        "components/**/__tests__/**",
        "hooks/**/__tests__/**",
        "lib/**/__tests__/**",
        "app/**", // Exclude Next.js app router pages
      ],
      thresholds: {
        global: {
          branches: 70,
          functions: 70,
          lines: 70,
          statements: 70,
        },
      },
    },
    css: true,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./"),
      "@/components": path.resolve(__dirname, "./components"),
      "@/lib": path.resolve(__dirname, "./lib"),
      "@/hooks": path.resolve(__dirname, "./hooks"),
      "@/types": path.resolve(__dirname, "./types"),
      "@/utils": path.resolve(__dirname, "./utils"),
      "@/stores": path.resolve(__dirname, "./stores"),
    },
  },
});
