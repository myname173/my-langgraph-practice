# src/agent/multimedia/state.py
from typing import TypedDict, List, Optional, Dict, Any


class MultimediaState(TypedDict):
    """
    顶层状态：管理整个长视频生成任务的全局流转
    """
    task: str
    global_setting: str
    scenes: List[Dict[str, Any]]
    current_scene_index: int
    final_movie_path: Optional[str]
    error_log: Optional[str]
    
    # 新增：是否启用首尾帧双控模式
    use_first_last_frame: bool

    # 一致性跟踪
    reference_images: List[str]
    reference_embeddings: List[List[float]]

    # 任务控制
    aborted: bool
    abort_reason: Optional[str]
