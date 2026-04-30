/**
 * 虚幻引擎 5 宣传页面 - Three.js 3D 场景
 * 负责渲染交互式 3D 背景和视觉效果
 */

// ============================================
// 全局变量
// ============================================
let scene, camera, renderer;
let geometry, material, mesh;
let mouseX = 0, mouseY = 0;
let windowHalfX, windowHalfY;
let clock;

// ============================================
// 初始化函数
// ============================================
function init() {
    // 获取 Hero 容器
    const heroContainer = document.getElementById('hero-canvas');
    if (!heroContainer) {
        console.error('Hero container not found');
        return;
    }

    // 创建场景
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.FogExp2(0x0a0a0a, 0.02);

    // 配置透视相机
    const fov = 75;
    const aspect = heroContainer.clientWidth / heroContainer.clientHeight;
    const near = 0.1;
    const far = 1000;
    camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    camera.position.z = 5;

    // 实例化 WebGL 渲染器
    renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true 
    });
    renderer.setSize(heroContainer.clientWidth, heroContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;

    // 将渲染器 DOM 元素添加到 Hero 容器
    heroContainer.appendChild(renderer.domElement);

    // 创建 3D 几何体模型
    createGeometry();

    // 添加光照
    setupLights();

    // 事件监听
    setupEventListeners(heroContainer);

    // 初始化时钟
    clock = new THREE.Clock();

    // 开始动画循环
    animate();
}

// ============================================
// 创建 3D 几何体
// ============================================
function createGeometry() {
    // 创建二十面体几何体（优化面数）
    geometry = new THREE.IcosahedronGeometry(1.5, 1);

    // 编写着色器材质代码以实现金属质感
    material = new THREE.MeshStandardMaterial({
        color: 0x00d4ff,
        metalness: 0.9,
        roughness: 0.2,
        envMapIntensity: 1.0,
        side: THREE.DoubleSide
    });

    // 创建网格
    mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);

    // 添加线框效果
    const wireframeGeometry = new THREE.WireframeGeometry(geometry);
    const wireframeMaterial = new THREE.LineBasicMaterial({ 
        color: 0x00d4ff,
        transparent: true,
        opacity: 0.3
    });
    const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
    mesh.add(wireframe);

    // 添加粒子系统
    createParticles();
}

// ============================================
// 创建粒子系统
// ============================================
function createParticles() {
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 500;
    
    const posArray = new Float32Array(particlesCount * 3);
    
    for(let i = 0; i < particlesCount * 3; i++) {
        posArray[i] = (Math.random() - 0.5) * 20;
    }
    
    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    
    const particlesMaterial = new THREE.PointsMaterial({
        size: 0.02,
        color: 0x00d4ff,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending
    });
    
    const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particlesMesh);
}

// ============================================
// 设置光照
// ============================================
function setupLights() {
    // 添加环境光照
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);

    // 添加动态点光源
    const pointLight1 = new THREE.PointLight(0x00d4ff, 1, 100);
    pointLight1.position.set(5, 5, 5);
    pointLight1.castShadow = true;
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xff0066, 0.5, 100);
    pointLight2.position.set(-5, -5, 5);
    scene.add(pointLight2);

    // 添加方向光
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);
}

// ============================================
// 设置事件监听
// ============================================
function setupEventListeners(container) {
    // 编写鼠标移动事件监听函数
    document.addEventListener('mousemove', onMouseMove);
    
    // 编写窗口大小调整时的相机适配逻辑
    window.addEventListener('resize', onWindowResize);
    
    // 编写滚动监听函数以触发视差效果
    window.addEventListener('scroll', onScroll);
}

// ============================================
// 鼠标移动事件处理
// ============================================
function onMouseMove(event) {
    mouseX = (event.clientX - windowHalfX) * 0.001;
    mouseY = (event.clientY - windowHalfY) * 0.001;
}

// ============================================
// 窗口大小调整处理
// ============================================
function onWindowResize() {
    const heroContainer = document.getElementById('hero-canvas');
    if (!heroContainer) return;

    windowHalfX = heroContainer.clientWidth / 2;
    windowHalfY = heroContainer.clientHeight / 2;

    camera.aspect = heroContainer.clientWidth / heroContainer.clientHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(heroContainer.clientWidth, heroContainer.clientHeight);
}

// ============================================
// 滚动事件处理
// ============================================
function onScroll() {
    const scrollY = window.scrollY;
    if (mesh) {
        mesh.rotation.y = scrollY * 0.002;
        mesh.position.y = scrollY * 0.001;
    }
}

// ============================================
// 动画渲染循环函数
// ============================================
function animate() {
    requestAnimationFrame(animate);

    const elapsedTime = clock.getElapsedTime();

    // 编写相机跟随鼠标运动的逻辑代码
    if (mesh) {
        mesh.rotation.x += 0.005;
        mesh.rotation.y += 0.005;
        
        // 鼠标交互
        mesh.rotation.x += mouseY * 0.5;
        mesh.rotation.y += mouseX * 0.5;
        
        // 浮动效果
        mesh.position.y += Math.sin(elapsedTime) * 0.002;
    }

    // 更新相机位置
    camera.position.x += (mouseX * 2 - camera.position.x) * 0.05;
    camera.position.y += (-mouseY * 2 - camera.position.y) * 0.05;
    camera.lookAt(scene.position);

    renderer.render(scene, camera);
}

// ============================================
// 页面加载完成后初始化
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    windowHalfX = window.innerWidth / 2;
    windowHalfY = window.innerHeight / 2;
    init();
});
