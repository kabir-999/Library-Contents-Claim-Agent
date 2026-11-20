import "./globals.css";
import type { ReactNode } from "react";
export const metadata = { title: "Library Contents Claim Agent" };
export default function RootLayout({ children }: { children: ReactNode }) {
  return (<html lang="en" suppressHydrationWarning><body suppressHydrationWarning><div className="mx-auto max-w-7xl p-4">{children}</div></body></html>);
}
