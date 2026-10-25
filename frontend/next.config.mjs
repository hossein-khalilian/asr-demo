import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Dev-only allowlist for /_next/* requests from hosts other than localhost
// (a port-forward, the deployed domain in the dev stack).
const allowedDevOrigins = ["localhost", "127.0.0.1", process.env.PUBLIC_URL]
  .map((value) => value?.trim())
  .filter((value) => value && value.length > 0);

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Self-contained server in .next/standalone: the prod image ships no node_modules.
  output: "standalone",
  // Do not let `next dev` write AGENTS.md into the repo.
  agentRules: false,
  allowedDevOrigins,
  turbopack: {
    root: __dirname,
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          // Recording needs the microphone on this origin only.
          { key: "Permissions-Policy", value: "camera=(), microphone=(self), geolocation=()" },
        ],
      },
    ];
  },
};

export default nextConfig;
