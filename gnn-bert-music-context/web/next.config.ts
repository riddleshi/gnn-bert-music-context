import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Preview / ingress hosts (Cursor CVM proxy) must be allowlisted or
  // Next.js 16 blocks /_next/* and the client never hydrates.
  allowedDevOrigins: [
    "127.0.0.1",
    "localhost",
    "*.localhost",
    "**.localhost",
    "*.agent.cvm.dev",
    "**.agent.cvm.dev",
    "*.cvm.dev",
    "**.cvm.dev",
    "*.cursor.sh",
    "**.cursor.sh",
    "*.cursor.com",
    "**.cursor.com",
  ],
  agentRules: false,
};

export default nextConfig;
