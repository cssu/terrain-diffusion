// Runs before every test file and clears the rendered page
// between tests so one test cannot affect the next

import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

afterEach(() => {
  cleanup()
})

// jsdom has no WebGL or ResizeObserver, so both are replaced with stand-ins

vi.mock('three', async (importOriginal) => {
  const three = await importOriginal<typeof import('three')>()

  class FakeWebGLRenderer {
    domElement = document.createElement('canvas')
    setPixelRatio() {}
    setSize(width: number, height: number) {
      this.domElement.width = width
      this.domElement.height = height
    }
    render() {}
    dispose() {}
  }

  return { ...three, WebGLRenderer: FakeWebGLRenderer }
})

class FakeResizeObserver {
  callback: () => void

  constructor(callback: () => void) {
    this.callback = callback
  }

  observe() {
    this.callback()
    window.addEventListener('resize', this.callback)
  }

  disconnect() {
    window.removeEventListener('resize', this.callback)
  }
}

vi.stubGlobal('ResizeObserver', FakeResizeObserver)
