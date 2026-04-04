import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  // Centralize env in project-root .env
  envDir: '..',
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/@ricky0123/vad-web/dist/vad.worklet.bundle.min.js',
          dest: './',
        },
        {
          src: 'node_modules/@ricky0123/vad-web/dist/silero_vad_legacy.onnx',
          dest: './',
        },
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: './',
        },
      ],
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/process': {
        target: 'https://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
      },
      '/search': {
        target: 'https://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
      },
      '/health': {
        target: 'https://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
      },
      '/chat': {
        target: 'https://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
      },
      '/chef/': {
        target: 'https://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
      },
      '/config': {
        target: 'https://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
      },
      '/ws/kitchen': {
        target: 'wss://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
        ws: true,
      },
      '/ws/chef-voice': {
        target: 'wss://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
        ws: true,
      },
      '/auth': {
        target: 'https://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
      },
      '/users': {
        target: 'https://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
      },
      '/api': {
        target: 'https://nutrisense-ai-production.up.railway.app',
        changeOrigin: true,
      },
    },
  },
})
