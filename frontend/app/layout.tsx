import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import { Providers } from "@/components/Providers";

export const metadata: Metadata = {
  title: "BowlIQ · fast-bowling biomechanics",
  description: "Upload a side-on clip of one delivery and get measured, ranked fixes with the frame each number came from.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <head>
        <script
          // Static string (no user data): reads the saved theme before first paint to avoid a light/dark flash.
          dangerouslySetInnerHTML={{ __html: "try{var s=JSON.parse(localStorage.getItem('bowliq.settings.v1')||'{}');if(s.theme==='light')document.documentElement.dataset.theme='light'}catch(e){}" }}
        />
      </head>
      <body data-theme="dark" suppressHydrationWarning>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
