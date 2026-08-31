/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output: the production image only needs
  // `.next/standalone` + `.next/static` + `public`, not the full
  // node_modules tree — see frontend/Dockerfile.
  output: "standalone",
  async rewrites() {
    // Server-side proxy target for /api/* — must be the Docker service
    // name (http://backend:8000) inside the container network; falls
    // back to localhost:8000 for native (non-Docker) `next dev`/`next start`.
    const backend = process.env.BACKEND_INTERNAL_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
