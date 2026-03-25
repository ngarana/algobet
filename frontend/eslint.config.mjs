import eslint from "@eslint/js";
import nextPlugin from "@next/eslint-plugin-next";
import jestPlugin from "eslint-plugin-jest";
import testingLibraryPlugin from "eslint-plugin-testing-library";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strict,
  {
    files: ["**/*.{ts,tsx}"],
    plugins: {
      "@next/next": nextPlugin,
      jest: jestPlugin,
      "testing-library": testingLibraryPlugin,
    },
    settings: {
      "testing-library": {
        "custom-module": "@testing-library/react",
      },
    },
    rules: {
      "@next/next/no-html-link-for-pages": "error",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "@typescript-eslint/no-explicit-any": "warn",
      "prefer-const": "error",
      "no-console": ["warn", { allow: ["warn", "error"] }],
      "jest/prefer-to-be": "warn",
      "jest/prefer-expect-assertions": "off",
      "testing-library/no-node-access": "warn",
      "testing-library/prefer-screen-queries": "warn",
    },
  },
  {
    files: ["**/*.test.{ts,tsx}", "**/*.spec.{ts,tsx}"],
    rules: {
      "testing-library/render-result-naming-convention": "off",
      "testing-library/no-unnecessary-act": "error",
    },
  },
  {
    ignores: ["node_modules/", ".next/", "out/", "coverage/"],
  }
);
