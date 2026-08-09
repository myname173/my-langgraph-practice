/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_LANGGRAPH_API_URL?: string;
  readonly VITE_LANGGRAPH_API_KEY?: string;
  readonly VITE_ASSISTANT_ID?: string;
  readonly VITE_MEDIA_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
