/**
 * 与后端 src/agent/multimedia 严格对齐的类型定义。
 * 所有 interrupt payload 结构均来自 graph.py 中 6 处 interrupt() 调用。
 */

/** 后端 _normalize_action 支持的动作 */
export type ReviewAction = "approve" | "rewrite" | "edit_prompt";

/** graph.py 中 6 个人审关卡的 stage 标识 */
export type InterruptStage =
  | "showrunner_review"
  | "prompt_preview"
  | "end_frame_prompt_preview"
  | "image_review"
  | "end_frame_review"
  | "video_review";

/** interrupt payload 的公共字段 */
interface BaseInterruptPayload {
  stage: InterruptStage;
  title: string;
  message: string;
  actions: ReviewAction[];
}

/** line 852: showrunner_review_gate_node —— 总导演人工审片 */
export interface ShowrunnerReviewPayload extends BaseInterruptPayload {
  stage: "showrunner_review";
  task: string;
  global_setting: string;
  scenes: SceneLike[];
}

/** line 1989: prompt_preview_node —— 生图前 prompt 审核 */
export interface PromptPreviewPayload extends BaseInterruptPayload {
  stage: "prompt_preview";
  scene_index: number;
  script: string;
  global_setting: string;
  image_prompt: string;
}

/** line 2042: end_frame_prompt_preview_node —— 尾帧 prompt 审核 */
export interface EndFramePromptPreviewPayload extends BaseInterruptPayload {
  stage: "end_frame_prompt_preview";
  scene_index: number;
  script: string;
  global_setting: string;
  image_prompt: string;
  reference_image_url: string;
}

/** line ~2317: image_review_gate_node —— 关键帧审片 */
export interface ImageReviewPayload extends BaseInterruptPayload {
  stage: "image_review";
  scene_index: number;
  script: string;
  global_setting: string;
  image_prompt: string;
  image_url: string;
  reference_image_url: string | null;
  embedding_similarity?: number | null;
  portrait_similarity?: number | null;
  continuity_score?: number | null;
  auto_feedback?: string;
}

/** line 2522: end_frame_review_gate_node —— 尾帧审片 */
export interface EndFrameReviewPayload extends BaseInterruptPayload {
  stage: "end_frame_review";
  scene_index: number;
  script: string;
  image_prompt: string;
  image_url: string;
  reference_image_url: string;
}

/** line ~2836: video_review_gate_node —— 视频审片 */
export interface VideoReviewPayload extends BaseInterruptPayload {
  stage: "video_review";
  scene_index: number;
  script: string;
  global_setting: string;
  video_prompt: string;
  video_url: string;
  reference_image_url: string | null;
  embedding_similarity?: number | null;
  auto_feedback?: string;
  action_beat?: string;
  motion_hint?: string;
}

export type InterruptPayload =
  | ShowrunnerReviewPayload
  | PromptPreviewPayload
  | EndFramePromptPreviewPayload
  | ImageReviewPayload
  | EndFrameReviewPayload
  | VideoReviewPayload;

/**
 * resume 时回传给后端的决策对象。
 * 后端 _normalize_decision 接受 dict，按 stage 读取对应可编辑字段。
 */
export interface ReviewDecision {
  action: ReviewAction;
  /** rewrite 时写入 critique / video_critique */
  reason?: string;
  /** showrunner_review 可改 */
  task?: string;
  global_setting?: string;
  scenes?: SceneLike[];
  /** prompt_preview / image_review / end_frame_* 可改 */
  image_prompt?: string;
  /** video_review 可改 */
  video_prompt?: string;
}

/** scenes[] 元素（后端为松散 dict，此处列出前端会用到的字段） */
export interface SceneLike {
  script?: string;
  image_prompt?: string;
  last_image_prompt?: string;
  video_prompt?: string;
  image_url?: string;
  last_image_url?: string;
  raw_video_url?: string;
  final_video_url?: string;
  action_beat?: string;
  critique?: string;
  video_critique?: string;
  is_perfect?: boolean;
  last_image_is_perfect?: boolean;
  video_is_perfect?: boolean;
  embedding_similarity?: number | null;
  portrait_similarity?: number | null;
  continuity_score?: number | null;
  [key: string]: unknown;
}

/** MultimediaState 中前端关注的子集（见 state.py） */
export interface MultimediaState {
  task?: string;
  global_setting?: string;
  scenes?: SceneLike[];
  current_scene_index?: number;
  final_movie_path?: string | null;
  final_movie_with_audio?: string | null;
  error_log?: string | null;
  aborted?: boolean;
  abort_reason?: string | null;
  visual_style?: string | null;
  style_description?: string | null;
  content_type?: string | null;
  character_portrait_url?: string | null;
  reference_images?: string[];
  rewrite_count?: number;
  studio_summary?: Record<string, unknown> | null;
  critic_eval?: Record<string, unknown> | null;
  quality_gates?: Record<string, unknown> | null;
  use_first_last_frame?: boolean;
  enable_audio?: boolean;
  voice_role?: string | null;
  bgm_mood?: string | null;
  [key: string]: unknown;
}

/** 启动一次 run 的输入（对齐 MultimediaState 必填项） */
export interface RunInput {
  task: string;
  global_setting: string;
  scenes: SceneLike[];
  current_scene_index: number;
  use_first_last_frame: boolean;
  enable_audio: boolean;
  voice_role?: string;
  bgm_mood?: string;
  reference_images: string[];
  reference_embeddings: number[][];
  aborted: boolean;
  final_movie_path: null;
  error_log: null;
  rewrite_count: number;
}

/** 节点执行时间线条目 */
export interface TimelineEntry {
  id: string;
  node: string;
  at: number;
  detail?: string;
}

export type RunStatus =
  | "idle"
  | "running"
  | "interrupted"
  | "done"
  | "error";
