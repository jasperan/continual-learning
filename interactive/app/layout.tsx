import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const viewport: Viewport = {
  themeColor: "#0a0b0f",
  width: "device-width",
  initialScale: 1,
};

export const metadata: Metadata = {
  title: "Continual Learning: Five Strategies for SLMs",
  description:
    "Teach a 1.5B parameter model new knowledge without forgetting. Interactive visualizations of DualMLP, TF-IDF gating, test-time training, retrieval-augmented learning, agentic context engineering, and hypernetwork-generated LoRA.",
  openGraph: {
    title: "Continual Learning for SLMs: Interactive Explorer",
    description:
      "Seven chapters, thirteen interactive widgets. Five strategies for teaching small language models new knowledge without catastrophic forgetting.",
    type: "website",
  },
  robots: "index, follow",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body suppressHydrationWarning className="min-h-full flex flex-col bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
