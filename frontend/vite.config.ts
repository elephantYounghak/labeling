import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/io': 'http://127.0.0.1:18080',
      '/project': {
        target: 'http://127.0.0.1:18080',
        bypass(req) {
          const url = req.url ?? ''
          if (url === '/projects' || url.startsWith('/projects/') || url.startsWith('/projects?')) {
            return '/index.html'
          }
        },
      },
      '/labeling': 'http://127.0.0.1:18080',
      '/ai': 'http://127.0.0.1:18080',
    },
  },
})
