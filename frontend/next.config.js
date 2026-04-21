/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  allowedDevOrigins: ["localhost", "127.0.0.1", "[::1]"],

  // API proxy configuration for Docker environment
  // This routes all /api/* requests to the backend service
  async rewrites() {
    // Determine the API URL based on environment
    // In Docker: use internal service name for server-side requests
    // On host: use localhost for browser-side requests
    const apiUrl = process.env.API_INTERNAL_URL || "http://localhost:8010";

    return [
      {
        source: "/api/:path*",
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },

  // Enable standalone output for production Docker builds
  output: "standalone",

  // Environment variables available at build time
  env: {
    API_INTERNAL_URL: process.env.API_INTERNAL_URL || "http://localhost:8010",
  },
};

module.exports = nextConfig;
