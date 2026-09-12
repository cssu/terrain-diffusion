import { useEffect, useRef } from 'react'
import { Mesh, MeshBasicMaterial, OrthographicCamera, PlaneGeometry, Scene, WebGLRenderer } from 'three'
import { frameCamera } from './camera'

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

    const plane = new Mesh(new PlaneGeometry(1, 1), new MeshBasicMaterial({ color: 'grey' }))
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
      plane.geometry.dispose()
      plane.material.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="terrain-view" />
}

export default TerrainView
