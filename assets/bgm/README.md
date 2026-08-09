# BGM 曲库

`generate_bgm(mood)` 从这里按情绪挑选背景音乐。

## 目录约定

```
assets/bgm/
├── ambient/     # 平静、中性、旁白向
├── tense/       # 紧张、悬疑
├── upbeat/      # 轻快、愉悦
├── epic/        # 史诗、宏大
├── romantic/    # 温柔、抒情
└── tech/        # 科技、电子
```

每个目录放任意数量的音频文件（`.mp3` / `.wav` / `.m4a` / `.ogg` / `.flac`）。
同一 mood 下有多首时随机选取；传 `seed` 可固定选择，便于复现调试。

目录为空或缺失时会回退到 `ambient/`，再为空则跳过配乐（不影响成片产出）。

## mood 从哪来

`graph.py` 的 `audio_mixer_node` 读取场景的 `emotion` 字段，
经 `resolve_bgm_mood()` 映射成上面 6 个 mood 之一，映射表见
`src/agent/multimedia/tools/audio.py` 的 `BGM_MOOD_BY_EMOTION`。

## 去哪找免版权音乐

- [Pixabay Music](https://pixabay.com/music/) — CC0，免注册可下载
- [Free Music Archive](https://freemusicarchive.org/) — 注意逐曲查看许可证
- [Incompetech](https://incompetech.com/music/royalty-free/) — CC-BY，需署名
- [YouTube Audio Library](https://studio.youtube.com/) — 需登录

建议每个 mood 放 1-3 首、每首 30 秒以上。
时长不足会自动循环铺满视频长度，并带 1s 淡入 / 1.5s 淡出。

## 自定义曲库位置

设置环境变量指向别处：

```powershell
$env:BGM_LIBRARY_DIR = "D:\my-bgm"
```

## 关于 AI 生成 BGM

当前不使用 AI 生成，原因记录在 `audio.py` 模块 docstring：

- dashscope 只有语音能力（ASR/TTS），**没有**音乐生成模型；
- audiocraft(MusicGen) 要求 Python <3.12，与本项目 3.13 环境不兼容，
  且 CPU 推理耗时长，会拖慢流水线迭代。

`generate_bgm()` 的签名是稳定的。将来若要换成 MusicGen 或第三方音乐 API，
只需替换该函数内部实现，上层调用方无需改动。
