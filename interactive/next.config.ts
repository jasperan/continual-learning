import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === "production";

const nextConfig: NextConfig = {
  output: "export",
  basePath: isProd ? "/continual-learning/interactive" : "",
  assetPrefix: isProd ? "/continual-learning/interactive/" : "",
  images: { unoptimized: true },
};

export default nextConfig;
