// 3D viewer: renders a MuJoCo sim peer in WebGL from mesh-streamed geometry.
//
// The server is authoritative: it runs the physics and streams (a) baked
// geometry once and (b) per-geom world poses (xpos/xmat) every frame. This
// module just builds Three.js meshes and slams the transforms each tick. No
// physics, no WASM, ~7KB/frame.
//
// Coordinate frames: MuJoCo is Z-up right-handed; Three.js is Y-up. We rotate
// the whole scene root -90deg about X so Z-up content sits correctly, rather
// than converting every geom matrix.

import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

export class Viewer3D {
  constructor(container) {
    this.container = container;
    this.geoms = [];        // index-aligned with model geoms; null = skipped
    this.meshObjects = [];   // THREE.Mesh per geom index (or null)
    this.ngeom = 0;
    this.ready = false;
    this._initThree();
  }

  _initThree() {
    const w = this.container.clientWidth || 640;
    const h = this.container.clientHeight || 480;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0e14);

    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
    this.camera.position.set(0.6, 0.5, 0.6);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setSize(w, h);
    this.renderer.shadowMap.enabled = true;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.target.set(0, 0.15, 0);

    // Lights
    const hemi = new THREE.HemisphereLight(0xffffff, 0x334155, 1.1);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.4);
    dir.position.set(1, 2, 1.5);
    dir.castShadow = true;
    this.scene.add(dir);

    // Z-up (MuJoCo) -> Y-up (Three) root.
    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2;
    this.scene.add(this.root);

    // Ground grid for spatial reference (in Three Y-up space, so add to scene).
    const grid = new THREE.GridHelper(2, 20, 0x334155, 0x1a212c);
    this.scene.add(grid);

    this._tmpMat = new THREE.Matrix4();

    window.addEventListener("resize", () => this._onResize());
    this._animate();
  }

  _onResize() {
    const w = this.container.clientWidth || 640;
    const h = this.container.clientHeight || 480;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  // Build the scene graph from a decoded geometry description.
  setGeometry(geo) {
    // Clear any existing meshes.
    for (const m of this.meshObjects) if (m) this.root.remove(m);
    this.meshObjects = [];
    this.ngeom = geo.ngeom || (geo.geoms ? geo.geoms.length : 0);
    this.geoms = geo.geoms || [];
    const meshes = geo.meshes || [];

    for (let i = 0; i < this.geoms.length; i++) {
      const g = this.geoms[i];
      if (!g) { this.meshObjects.push(null); continue; }       // skipped (collision)
      if (g.group === 3) { this.meshObjects.push(null); continue; } // hide collision

      let geometry = null;
      if (g.type === "mesh" && g.mesh != null && meshes[g.mesh]) {
        geometry = this._buildMeshGeometry(meshes[g.mesh]);
      } else if (g.type === "box") {
        const s = g.size;
        geometry = new THREE.BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2);
      } else if (g.type === "sphere") {
        geometry = new THREE.SphereGeometry(g.size[0], 24, 16);
      } else if (g.type === "capsule") {
        geometry = new THREE.CapsuleGeometry(g.size[0], g.size[1] * 2, 8, 16);
      } else if (g.type === "cylinder") {
        geometry = new THREE.CylinderGeometry(g.size[0], g.size[0], g.size[1] * 2, 24);
        geometry.rotateX(Math.PI / 2); // MuJoCo cylinder axis is Z
      } else if (g.type === "ellipsoid") {
        geometry = new THREE.SphereGeometry(1, 24, 16);
        geometry.scale(g.size[0], g.size[1], g.size[2]);
      } else if (g.type === "plane") {
        // Skip planes in the WebGL view; the GridHelper is our floor.
        this.meshObjects.push(null);
        continue;
      }

      if (!geometry) { this.meshObjects.push(null); continue; }

      const rgba = g.rgba || [0.7, 0.7, 0.7, 1];
      const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
      const mat = new THREE.MeshStandardMaterial({
        color,
        metalness: 0.1,
        roughness: 0.7,
        transparent: (rgba[3] != null && rgba[3] < 1),
        opacity: rgba[3] != null ? rgba[3] : 1,
      });
      const mesh = new THREE.Mesh(geometry, mat);
      mesh.matrixAutoUpdate = false; // we set matrix directly from xpos/xmat
      this.root.add(mesh);
      this.meshObjects.push(mesh);
    }
    this.ready = true;
  }

  _buildMeshGeometry(m) {
    const g = new THREE.BufferGeometry();
    const verts = new Float32Array(m.vert);
    g.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    if (m.face && m.face.length) {
      const idx = m.face.length > 65535
        ? new Uint32Array(m.face)
        : new Uint16Array(m.face);
      g.setIndex(new THREE.BufferAttribute(idx, 1));
    }
    g.computeVertexNormals();
    return g;
  }

  // Apply one pose frame: xpos (ngeom*3), xmat (ngeom*9, row-major 3x3).
  setPose(frame) {
    if (!this.ready) return;
    const xpos = frame.xpos, xmat = frame.xmat;
    if (!xpos || !xmat) return;
    const n = Math.min(this.meshObjects.length, xpos.length / 3);
    for (let i = 0; i < n; i++) {
      const mesh = this.meshObjects[i];
      if (!mesh) continue;
      const p = i * 3, r = i * 9;
      // MuJoCo xmat is row-major 3x3. Three Matrix4.set is row-major too.
      this._tmpMat.set(
        xmat[r + 0], xmat[r + 1], xmat[r + 2], xpos[p + 0],
        xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
        xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
        0, 0, 0, 1
      );
      mesh.matrix.copy(this._tmpMat);
    }
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}

// Decode a gzip+base64 geometry payload using the browser's DecompressionStream.
export async function decodeGeometry(payload) {
  if (payload.encoding !== "gzip+b64") {
    throw new Error("unknown geometry encoding: " + payload.encoding);
  }
  const bin = Uint8Array.from(atob(payload.data), c => c.charCodeAt(0));
  const ds = new DecompressionStream("gzip");
  const stream = new Blob([bin]).stream().pipeThrough(ds);
  const text = await new Response(stream).text();
  return JSON.parse(text);
}
