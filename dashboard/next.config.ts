import type { NextConfig } from "next";
import { config as loadEnv } from "dotenv";
import { resolve } from "path";

// Load the single root .env so dashboard/.env.local is not needed.
// Works for both `next dev` and `next build`.
loadEnv({ path: resolve(__dirname, "../.env") });

const nextConfig: NextConfig = {};

export default nextConfig;
