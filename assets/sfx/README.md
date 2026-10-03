# SFX 音效曲库（P1-5）

把免版权音效按**类别**放入子目录即可启用转场音效；目录为空或不存在时自动跳过（不影响成片）。

支持类别（转场类型 → 回退链）：

| 转场类型 | 优先查找 | 回退 |
|---|---|---|
| smash_cut | `impact/` | `hit/` |
| whip_pan / camera_carry | `whoosh/` | `swish/` |
| dissolve | `soft/` | `transition/` |
| fade_through_black | `deep/` | `transition/` |
| hard_cut | 不铺音效 | — |

格式：`.mp3 / .wav / .m4a / .ogg / .flac`，时长建议 0.3~1.5s。
默认音量 0.6，可用环境变量 `SFX_LIBRARY_DIR` 指向自定义曲库。

示例结构：

```
assets/sfx/
├── impact/
│   └── hit_low.mp3
├── whoosh/
│   └── swish_fast.wav
└── soft/
    └── transition airy.mp3
```
