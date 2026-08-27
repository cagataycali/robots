import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

/**
 * Caching policy is a *safety* decision here, not a performance one.
 *
 * This app moves real motors. Anything that commands a robot must be
 * `NetworkOnly` with no Background Sync: a queued POST that replays when
 * connectivity returns would start a task minutes later, at an arm nobody is
 * watching. Only descriptive, idempotent GETs are allowed a cache, and only
 * NetworkFirst so a stale answer is a fallback rather than the default.
 */
const ACTUATING = /\/api\/(robots\/[^/]+\/(task|stop|twin)|safety\/|devices\/(spawn|despawn)|mesh\/(restart|config)|agent\/reset|config)/

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      // `prompt`, not `autoUpdate`: an automatic reload mid-task tears down the
      // camera sockets and the run form of a robot that is currently moving.
      registerType: 'prompt',
      includeAssets: ['icon.svg', 'apple-touch-icon.png'],
      manifest: {
        id: '/',
        name: 'Strands Robots Dashboard',
        short_name: 'Robots',
        description: 'Fleet cockpit for the strands-robots mesh',
        theme_color: '#0a0e14',
        background_color: '#0a0e14',
        display: 'standalone',
        display_override: ['window-controls-overlay', 'standalone', 'browser'],
        orientation: 'any',
        scope: '/',
        start_url: '/',
        categories: ['productivity', 'utilities'],
        icons: [
          { src: '/icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' },
          { src: '/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' },
          { src: '/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' },
          { src: '/maskable-192.png', sizes: '192x192', type: 'image/png', purpose: 'maskable' },
          { src: '/maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
        // Deliberately no e-stop shortcut: a long-press menu is exactly where a
        // fleet-wide stop must NOT be, and the sheet needs a confirmation step
        // plus per-peer results that a shortcut cannot show.
        shortcuts: [
          { name: 'Fleet', short_name: 'Fleet', url: '/', description: 'All robots on the mesh' },
          { name: 'Ask the agent', short_name: 'Agent', url: '/?panel=chat', description: 'Talk to the fleet agent' },
          { name: 'Settings', short_name: 'Settings', url: '/?panel=settings', description: 'Backend, agent and mesh configuration' },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,png,woff2}'],
        navigateFallback: '/index.html',
        // Never let the SW answer for the API host when it is a different origin.
        navigateFallbackDenylist: [/^\/api/, /^\/ws/],
        cleanupOutdatedCaches: true,
        clientsClaim: true,
        runtimeCaching: [
          {
            // The policy catalog and the robot registry are the run form's
            // schema: read-only, slow-changing, and useless to stale-block.
            urlPattern: /\/api\/(policies|robots\/registry)$/,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'strands-schema',
              networkTimeoutSeconds: 4,
              expiration: { maxEntries: 8, maxAgeSeconds: 60 * 60 * 24 },
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            // Everything that can move a robot, plus config writes.
            urlPattern: ACTUATING,
            handler: 'NetworkOnly',
          },
          {
            // Live state must never be served from a cache - a cached fleet
            // snapshot shows a robot as idle while it is mid-task.
            urlPattern: /\/api\//,
            handler: 'NetworkOnly',
          },
        ],
      },
      devOptions: { enabled: false },
    }),
  ],
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
      '/ws': { target: 'ws://localhost:8080', ws: true },
    },
  },
})
