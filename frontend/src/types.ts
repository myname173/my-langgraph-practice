/**
 * 与后端 src/agent/multimedia 严格对齐的类型定义。
 * 所有 interrupt payload 结构均来自 graph.py 中 6 处 interrupt() 调用。
 */

/** 后端 _normalize_action 支持的动作 */
export type ReviewAction = "approve" | "rewrite" | "edit_prompt";

/** graph.py 中人工审核关卡的 stage 标识 */
export type InterruptStage =
  | "showrunner_review"
  | "prompt_preview"
  | "end_frame_prompt_preview"
  | "image_review"
  | "end_frame_review"
  | "video_review"
  | "draft_review";

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

/** 成片草稿审片（stitcher 之后、audio_mixer 之前的无声粗剪确认） */
export interface DraftReviewPayload extends BaseInterruptPayload {
  stage: "draft_review";
  video_url: string;
}

export type InterruptPayload =
  | ShowrunnerReviewPayload
  | PromptPreviewPayload
  | EndFramePromptPreviewPayload
  | ImageReviewPayload
  | EndFrameReviewPayload
  | VideoReviewPayload
  | DraftReviewPayload;

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
  /** stitcher 回写的真实片段时长（秒），即该镜头视频实际长度 */
  video_duration?: number | null;
  [key: string]: unknown;
}

/** 单条字幕条目（后端 parse_srt_to_entries 产出，对齐 subtitle.py） */
export interface SubtitleEntry {
  index: number;
  /** 起始时间（秒） */
  start: number;
  /** 结束时间（秒） */
  end: number;
  /** 字幕文本（已去除换行） */
  text: string;
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
  /** 一致性参考图：{ characters:{name:{url,description,name_cn}}, props:{...}, environments:{...} } */
  reference_sheets?: Record<string, unknown> | null;
  /** 素材匹配可见化报告：哪些用户参考图被剧本用到、命中几个镜头（见 AssetMatchReport） */
  asset_match_report?: AssetMatchReport | null;
  rewrite_count?: number;
  studio_summary?: Record<string, unknown> | null;
  critic_eval?: Record<string, unknown> | null;
  quality_gates?: Record<string, unknown> | null;
  use_first_last_frame?: boolean;
  enable_audio?: boolean;
  voice_role?: string | null;
  bgm_mood?: string | null;
  prefer_native_audio?: boolean;
  native_audio_kept?: boolean;
  /** 解析后的字幕条目（逐句时间轴），供前端「字幕生成在第几秒」展示 */
  subtitle_entries?: SubtitleEntry[] | null;
  /** 配音总时长（秒） */
  voiceover_duration?: number | null;
  /** 音频闭环状态：native(原生音轨优先) / dubbed(本地配音混音) / 未跑 */
  audio_status?: "native" | "dubbed" | null;
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
  prefer_native_audio?: boolean;
  reference_images: string[];
  reference_embeddings: number[][];
  /** 用户上传/勾选的素材参考图（角色/道具/场景），作为一致性参考注入生图与视频流程；
   *  这些图本身不会直接生成视频，而是指导生成内容的角色/道具/场景一致性。 */
  reference_sheets?: ReferenceSheets;
  /** 是否携带外部素材（上传或勾选），置 true 后后端 reference_gen_node 跳过重生图、直接复用。 */
  assets_imported?: boolean;
  /** 低干预 / 自动模式：为真时后端跳过 6 道人工审核闸口，直接以 approve 放行。 */
  auto_mode?: boolean;
  /** 所属故事标识（如 cyber/gull/xianxia），用于素材库按故事隔离；为空时前端回退到 threadId 推断。 */
  story?: string;
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

/**
 * 单条参考图资产（与后端 reference_gen._persist_reference_image 同 schema）。
 * 一个资产名下可挂多张参考图（如"女主"含正面/侧脸/全身），用于多图融合注入续跑生图。
 */
export interface ReferenceEntry {
  /** 兼容旧单 url：这里统一用数组，单图时长度为 1。 */
  urls: string[];
  description: string;
  name: string;
  /** 所属故事标签（如 xianxia），用于素材来源归类与展示，不影响后端锚定逻辑 */
  story?: string;
}

/** reference_sheets 结构：角色 / 道具 / 环境三类参考图 */
export interface ReferenceSheets {
  characters?: Record<string, ReferenceEntry>;
  props?: Record<string, ReferenceEntry>;
  environments?: Record<string, ReferenceEntry>;
}

/**
 * 素材匹配可见化报告（后端 asset_match_report）：
 *   matched:    { 资产名: 命中镜头数 } —— 该资产名至少在某镜头脚本中出现，并实际参与生图
 *   unmatched:  string[]            —— 在所有镜头脚本中都搜不到的资产名（极可能静默失效）
 * 用于给每张用户参考图标「已用于 N 个镜头 / 未命中剧本」徽标，消除名称对不上导致的静默失效。
 */
export interface AssetMatchReport {
  matched?: Record<string, number>;
  unmatched?: string[];
}

/** 素材库分类：角色(人物) / 武器(道具) / 场景(环境) / 忽略(不纳入) */
export type AssetCategory = "character" | "prop" | "environment" | "ignore";

/** 一个归一化的资产 url（兼容 ReferenceEntry.url 可能是 string 或 string[]） */
export type ReferenceUrl = string | string[];

/** 把 ReferenceEntry.url（string | string[] | undefined）归一化为字符串数组 */
export function normalizeUrls(url: ReferenceUrl | undefined): string[] {
  if (!url) return [];
  return Array.isArray(url) ? url.filter(Boolean) : [url];
}

/** 一张素材图的展示与可编辑状态 */
export interface AssetItem {
  filename: string;
  url: string;
  category: AssetCategory;
  label: string;
  description?: string;
  role_name?: string;
  /** 所属故事（cyber / gull），仅用于历史图分组提示 */
  story?: string;
}

/** 按 role_name 聚合后的资产卡（前端展示用一等实体） */
export interface AssetGroup {
  /** 资产名 = role_name */
  name: string;
  category: Exclude<AssetCategory, "ignore">;
  /** Persona 设定 / 描述（来自 manifest.description 或用户编辑） */
  description?: string;
  /** 该资产下所有参考图 */
  items: AssetItem[];
  /** 所属故事（cyber / gull） */
  story?: string;
}

/** 前端保存素材库时传给 POST /assets/apply 的载荷 */
export interface AssetsApplyPayload {
  thread_id: string;
  reference_sheets: ReferenceSheets;
}

/** 一张已选素材（含本地上传与勾选的已有图）。图本身只作为一致性参考，不生成视频。 */
export interface SelectedAsset {
  /** 可访问的图 URL（/media/... 或上传后返回的 url） */
  url: string;
  /** 展示名（文件名或资产名） */
  name: string;
  /** 素材分类：角色 / 道具 / 场景（不含 ignore）。上传图在用户未明确选择前为空串，
   * 提交时校验，避免道具/场景图被默认误归为「角色」。 */
  category: Exclude<AssetCategory, "ignore"> | "";
  /** 可选描述，拼入 reference_sheets 的 description */
  description?: string;
  /** 来源标记，便于前端区分上传图与已有图 */
  source: "upload" | "library";
  /** 故事标签（如 xianxia），继承自建任务时的 story，用于把素材归类为对应故事 */
  story?: string;
}

/** 新建任务时收集到的素材选择（上传 + 勾选已有），用于构建 ReferenceSheets。 */
export interface AssetSelection {
  /** 已选素材平铺列表 */
  items: SelectedAsset[];
}
