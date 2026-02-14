import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { Providers } from './providers'
import { Navbar, Sidebar, Breadcrumb } from '@/components/layout'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AlgoBet - Football Match Predictions',
  description: 'AI-powered football match predictions and analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <div className="min-h-screen bg-background">
            <Navbar />
            <Sidebar />
            <main className="md:pl-64">
              <div className="container mx-auto p-4">
                <div className="mb-4">
                  <Breadcrumb />
                </div>
                {children}
              </div>
            </main>
          </div>
        </Providers>
      </body>
    </html>
  )
}
