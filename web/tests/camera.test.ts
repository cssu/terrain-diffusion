import { OrthographicCamera } from 'three'
import { expect, it } from 'vitest'
import { frameCamera } from '../src/camera'

it.each([
  [800, 400],
  [400, 800],
  [500, 500],
])('shows the whole plane unstretched in a %ix%i window', (width, height) => {
  const camera = new OrthographicCamera()

  frameCamera(camera, width, height)

  const viewWidth = camera.right - camera.left
  const viewHeight = camera.top - camera.bottom
  expect(Math.min(viewWidth, viewHeight)).toBeCloseTo(1)
  expect(viewWidth / viewHeight).toBeCloseTo(width / height)
})
