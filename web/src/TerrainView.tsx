import { useEffect, useRef } from 'react'
import { Mesh, MeshBasicMaterial, OrthographicCamera, PlaneGeometry, Scene, WebGLRenderer } from 'three'
import { frameCamera } from './camera'
import { heightTexture, sampleHeights } from './terrain'

const GRID_SIZE = 32

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

    const texture = heightTexture(sampleHeights(GRID_SIZE), GRID_SIZE, GRID_SIZE)
    const plane = new Mesh(new PlaneGeometry(1, 1), new MeshBasicMaterial({ map: texture }))
    scene.add(plane)

    const observer = new ResizeObserver(() => {
      const { clientWidth: width, clientHeight: height } = container
      renderer.setSize(width, height)
      frameCamera(camera, width, height)
      renderer.render(scene, camera)
    })
    observer.observe(container)

    return () => {
      observer.disconnect()
      texture.dispose()
      plane.geometry.dispose()
      plane.material.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="terrain-view" />
}

export default TerrainView
