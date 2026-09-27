import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * The bundle is served by the Python dashboard process: `server.py` mounts the
 * `static/` directory at `/static` and answers `/` with `static/index.html`.
 * `base: '/static/'` makes every asset URL resolve under that mount, so the
 * server needs no `/assets` route; `outDir: '../static'` writes straight into
 * the package so the wheel ships the built UI and nothing runs node at runtime.
 * `emptyOutDir: false` keeps what the build does not own: the three.js vendor
 * files and `twin.js`, which the Sim tab loads as a module script.
 *
 * No service worker on purpose. This page moves motors; a cached shell or a
 * replayed request is a liability, and under `/static/` a worker could not
 * control `/` anyway.
 */
export default defineConfig({
  plugins: [react()],
  base: '/static/',
  publicDir: 'public',
  build: {
    outDir: '../static',
    emptyOutDir: false,
    sourcemap: false,
    // Fixed names, so a rebuild overwrites the previous bundle instead of
    // leaving a hashed sibling behind in a directory the build does not empty.
    rollupOptions: {
      output: {
        entryFileNames: 'assets/index.js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name][extname]',
      },
    },
    chunkSizeWarningLimit: 600,
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
      '/ws': { target: 'ws://localhost:8080', ws: true },
      '/static/twin.js': 'http://localhost:8080',
      '/static/vendor': 'http://localhost:8080',
    },
  },
})
