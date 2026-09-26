import { render } from '@testing-library/react'
import { expect, it } from 'vitest'
import TerrainView from '../src/TerrainView'

function setSize(element: Element, width: number, height: number) {
  Object.defineProperty(element, 'clientWidth', { value: width, configurable: true })
  Object.defineProperty(element, 'clientHeight', { value: height, configurable: true })
}

it('resizes the canvas with the window', () => {
  const { container } = render(<TerrainView />)
  const view = container.firstElementChild!

  setSize(view, 800, 600)
  window.dispatchEvent(new Event('resize'))

  const canvas = view.querySelector('canvas')!
  expect([canvas.width, canvas.height]).toEqual([800, 600])
})

it('removes the canvas when it leaves the page', () => {
  const { container, unmount } = render(<TerrainView />)
  const view = container.firstElementChild!

  unmount()

  expect(view.querySelector('canvas')).toBeNull()
})
