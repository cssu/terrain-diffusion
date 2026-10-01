import { useEffect, useRef } from 'react'
import {
  type DataTexture,
  Mesh,
  MeshBasicMaterial,
  OrthographicCamera,
  PlaneGeometry,
  Scene,
  WebGLRenderer,
} from 'three'
import { frameCamera } from './camera'
import { heightTexture } from './terrain'
import { requestTerrain, type TerrainRequest } from './terrainSource'

const REGION: TerrainRequest = { seed: 1, row: 0, column: 0, rows: 32, columns: 32 }

function TerrainView() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current!
    const renderer = new WebGLRenderer()
    renderer.setPixelRatio(window.devicePixelRatio)
    container.appendChild(renderer.domElement)

    const scene = new Scene()
    const camera = new OrthographicCamera()
    camera.position.z = 1

    const plane = new Mesh(new PlaneGeometry(1, 1), new MeshBasicMaterial())
    scene.add(plane)

    const draw = () => {
      const { clientWidth: width, clientHeight: height } = container
      renderer.setSize(width, height)
      frameCamera(camera, width, height)
      renderer.render(scene, camera)
    }

    const observer = new ResizeObserver(draw)
    observer.observe(container)

    let texture: DataTexture | undefined
    let showing = true

    requestTerrain(REGION).then(({ request, heights }) => {
      if (!showing) return

      texture = heightTexture(heights, request.columns, request.rows)
      plane.material.map = texture
      plane.material.needsUpdate = true
      draw()
    })

    return () => {
      showing = false
      observer.disconnect()
      texture?.dispose()
      plane.geometry.dispose()
      plane.material.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="terrain-view" />
}

export default TerrainView
