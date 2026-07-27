import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    include: ['tests/**/*.test.{ts,tsx}'],
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    reporters: process.env.JUNIT_XML
      ? ['default', ['junit', { outputFile: process.env.JUNIT_XML }]]
      : ['default'],
  },
})
