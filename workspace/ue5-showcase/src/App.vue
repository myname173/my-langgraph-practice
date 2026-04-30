<template>
  <div id="app">
    <canvas ref="canvas3d"></canvas>
    <div class="ui-overlay">
      <header class="header">
        <h1>UE5 Showcase</h1>
        <p class="subtitle">Unreal Engine 5 Interactive Demo</p>
      </header>
      <nav class="scene-nav">
        <button 
          v-for="(scene, index) in scenes" 
          :key="index"
          @click="loadScene(index)"
          :class="{ active: currentScene === index }"
        >
          {{ scene.name }}
        </button>
      </nav>
      <div class="controls">
        <label>
          <input type="checkbox" v-model="autoRotate" />
          Auto Rotate
        </label>
        <label>
          <input type="checkbox" v-model="showWireframe" />
          Wireframe
        </label>
      </div>
      <footer class="footer">
        <p>Built with Three.js & Vue</p>
      </footer>
    </div>
    <div class="loading" v-if="loading">
      <div class="spinner"></div>
      <p>Loading...</p>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as THREE from 'three'

export default {
  name: 'App',
  setup() {
    const canvas3d = ref(null)
    const loading = ref(true)
    const currentScene = ref(0)
    const autoRotate = ref(true)
    const showWireframe = ref(false)
    
    const scenes = ref([
      { name: 'Cube', module: null },
      { name: 'Sphere', module: null },
      { name: 'Torus', module: null }
    ])

    let scene = null
    let camera = null
    let renderer = null
    let animationId = null
    let currentSceneObject = null

    const initThree = () => {
      const canvas = canvas3d.value
      const width = window.innerWidth
      const height = window.innerHeight

      scene = new THREE.Scene()
      scene.background = new THREE.Color(0x1a1a2e)

      camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
      camera.position.z = 5

      renderer = new THREE.WebGLRenderer({ 
        canvas, 
        antialias: true,
        alpha: true 
      })
      renderer.setSize(width, height)
      renderer.setPixelRatio(window.devicePixelRatio)

      const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
      scene.add(ambientLight)

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
      directionalLight.position.set(5, 5, 5)
      scene.add(directionalLight)

      loadScene(0)
      loading.value = false
    }

    const loadScene = (index) => {
      if (currentSceneObject) {
        scene.remove(currentSceneObject)
      }

      currentScene.value = index
      
      const geometry = new THREE.BoxGeometry(2, 2, 2)
      const material = new THREE.MeshStandardMaterial({ 
        color: 0x00d4ff,
        wireframe: showWireframe.value
      })
      currentSceneObject = new THREE.Mesh(geometry, material)
      scene.add(currentSceneObject)
    }

    const animate = () => {
      animationId = requestAnimationFrame(animate)

      if (autoRotate.value && currentSceneObject) {
        currentSceneObject.rotation.x += 0.01
        currentSceneObject.rotation.y += 0.01
      }

      renderer.render(scene, camera)
    }

    const handleResize = () => {
      const width = window.innerWidth
      const height = window.innerHeight

      camera.aspect = width / height
      camera.updateProjectionMatrix()

      renderer.setSize(width, height)
    }

    watch(showWireframe, (newValue) => {
      if (currentSceneObject) {
        currentSceneObject.material.wireframe = newValue
      }
    })

    onMounted(() => {
      initThree()
      animate()
      window.addEventListener('resize', handleResize)
    })

    onUnmounted(() => {
      if (animationId) {
        cancelAnimationFrame(animationId)
      }
      window.removeEventListener('resize', handleResize)
      if (renderer) {
        renderer.dispose()
      }
    })

    return {
      canvas3d,
      loading,
      scenes,
      currentScene,
      autoRotate,
      showWireframe,
      loadScene
    }
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  width: 100vw;
  height: 100vh;
  position: relative;
  overflow: hidden;
}

#app canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.ui-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 20px;
}

.header {
  text-align: center;
  color: white;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
}

.header h1 {
  font-size: 3rem;
  font-weight: 700;
  margin-bottom: 10px;
  background: linear-gradient(135deg, #00d4ff, #7b2cbf);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.subtitle {
  font-size: 1.2rem;
  opacity: 0.8;
}

.scene-nav {
  pointer-events: auto;
  display: flex;
  justify-content: center;
  gap: 10px;
  margin: 20px 0;
}

.scene-nav button {
  padding: 10px 20px;
  border: 2px solid #00d4ff;
  background: rgba(0, 212, 255, 0.1);
  color: #00d4ff;
  border-radius: 25px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;
}

.scene-nav button:hover {
  background: rgba(0, 212, 255, 0.3);
  transform: translateY(-2px);
}

.scene-nav button.active {
  background: #00d4ff;
  color: #1a1a2e;
}

.controls {
  pointer-events: auto;
  display: flex;
  justify-content: center;
  gap: 20px;
  color: white;
}

.controls label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.controls input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.footer {
  text-align: center;
  color: white;
  opacity: 0.6;
  font-size: 0.9rem;
}

.loading {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: #1a1a2e;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: white;
  z-index: 1000;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 4px solid rgba(0, 212, 255, 0.3);
  border-top-color: #00d4ff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 20px;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
