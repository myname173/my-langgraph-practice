import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// LangGraph Server 默认端口 2024（langgraph dev），可用 env 覆盖。
const LANGGRAPH_TARGET =
  process.env.VITE_LANGGRAPH_PROXY_TARGET ?? "http://127.0.0.1:2024";
// 产物静态服务（src/agent/multimedia/static_server.py）
const MEDIA_TARGET =
  process.env.VITE_MEDIA_PROXY_TARGET ?? "http://127.0.0.1:8900";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 把 LangGraph Server 代理到同源，规避 CORS 与 SSE 跨域问题
      "/lg": {
        target: LANGGRAPH_TARGET,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/lg/, ""),
      },
      // 产物（图片 / 视频 / 音频）静态托管
      "/media": {
        target: MEDIA_TARGET,
        changeOrigin: true,
      },
      // 历史会话恢复接口（src/agent/multimedia/history_api.py），与 /media 同源转发
      "/history": {
        target: MEDIA_TARGET,
        changeOrigin: true,
      },
      // 素材上传 / 应用接口（src/agent/multimedia/static_server.py 的 assets_router），与 /media 同源转发
      "/assets": {
        target: MEDIA_TARGET,
        changeOrigin: true,
      },
      // 快速生成接口（static_server.py 的 POST /generate），与 /media 同源转发
      "/generate": {
        target: MEDIA_TARGET,
        changeOrigin: true,
      },
      // 设置面板接口（static_server.py 的 GET/POST /settings），与 /media 同源转发
      "/settings": {
        target: MEDIA_TARGET,
        changeOrigin: true,
      },
      // 后端健康检查（static_server.py 的 GET /health），同源转发，供开场动画探测
      "/healthz": {
        target: MEDIA_TARGET,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/healthz/, "/health"),
      },
    },
  },
});
