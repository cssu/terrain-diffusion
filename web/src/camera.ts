import type { OrthographicCamera } from 'three'

export function frameCamera(camera: OrthographicCamera, width: number, height: number) {
  const aspect = width / height
  const halfHeight = aspect >= 1 ? 0.5 : 0.5 / aspect

  camera.left = -halfHeight * aspect
  camera.right = halfHeight * aspect
  camera.top = halfHeight
  camera.bottom = -halfHeight
  camera.updateProjectionMatrix()
}
