import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      "/models": {
        target: "http://localhost:5174/", // Python backend
        changeOrigin: true,
        secure: false,
      },
      "/predict": {
        target: "http://localhost:5174/", // Python backend
        changeOrigin: true,
        secure: false,
      }
    },
    allowedHosts: ['f0c832e00caa.ngrok.app']
  }
})
