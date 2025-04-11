// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      // Properly handle dynamic imports from onnxruntime-web
      external: [/^onnxruntime-web\/dist\/.*/],
    }
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  }
});