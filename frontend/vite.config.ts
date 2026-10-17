import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/io': 'http://127.0.0.1:18080',
      '/project': 'http://127.0.0.1:18080',
      '/labeling': 'http://127.0.0.1:18080',
      '/ai': 'http://127.0.0.1:18080',
    },
  },
})
