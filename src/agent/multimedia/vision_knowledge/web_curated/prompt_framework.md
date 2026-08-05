# AI 视频 / 图像 Prompt 工程方法论

> 来源：CSDN AI视频提示词全攻略、GitHub ai-shortfilm-prompts、Microsoft Sora Case Studies、B站/掘金技术社区。
> 用途：构建高质量 AI 生成内容的 prompt 框架和最佳实践。

---

## 7 模块 Prompt 构建法

来自 CSDN 社区总结的高质量 AI 视频 prompt 核心逻辑：

| 模块 | 说明 | 示例 |
|---|---|---|
| 风格 | 整体视觉基调和画面风格 | "电影级人像，温柔治愈，色调柔和" |
| 视角/运镜 | 摄影机运动方式和角度 | "中景+缓慢推镜，平稳跟拍，浅景深背景虚化" |
| 主体 | 画面核心内容描述 | "一位时尚女性漫步在霓虹闪烁的街头" |
| 动作 | 动态行为的具体描述 | "角色缓慢旋转360度，展示细节" |
| 光影 | 照明方案和光效描述 | "霓虹冷光，蓝紫撞色光线，明暗对比强烈" |
| 质量词 | 画质和细节提升标签 | "cinematic lighting, 8K, highly detailed" |
| 约束 | 限制条件和负面指令 | "人物五官清晰不扭曲，画面稳定无水印" |

---

## 5 层权重结构 (剧本式 Prompt)

适用于 AI 短剧 / 叙事型视频 prompt 的权重分配：

```
[角色设定 15%] A battle-scarred veteran soldier, cybernetic left arm, weary expression
[场景环境 15%] Abandoned military outpost in arctic tundra, blizzard conditions, rusted equipment
[动作序列 40%] The soldier slowly stands from a frozen command chair, activates a holographic map display, traces a route with a trembling finger, then holsters a sidearm and pushes through the blast door into the storm
[对话交互 15%] Mutters "Last chance..." in English, speaks firmly but with underlying exhaustion
[风格标签 15%] Cinematic, ARRI Alexa look, desaturated cool palette, film grain, anamorphic lens flare
```

### 动作描写递进逻辑

描述动态画面时遵循：**基础 → 细节 → 连续 → 结果**

- **基础**：核心动作（stands up）
- **细节**：动作方式（slowly, with difficulty, joints cracking）
- **连续**：后续行为链（activates display, traces route, holsters weapon）
- **结果**：动作产生的视觉/叙事效果（pushes into storm, silhouette against blizzard）

---

## 硬件锚定法 (Hardware Specificity)

来自 GitHub ai-shortfilm-prompts 的核心技巧：

不要请求"cinematic feel"这种泛化描述，而是引用真实的光学设备来锚定生成的画面美学。

### 推荐设备参考

| 设备 | 画面特征 | 适用场景 |
|---|---|---|
| ARRI Alexa 65 | 大画幅、浅景深、丰富暗部细节 | 高端电影、人物特写 |
| IMAX 70mm | 超高分辨率、极低颗粒、壮观广角 | 史诗场景、太空、自然 |
| Panavision C-series Anamorphic | 经典变形宽银幕、水平光晕、椭圆虚化 | 70s-80s 复古电影感 |
| RED DSMC2 | 锐利数字感、高帧率能力 | 动作、运动镜头 |
| Sony Venice | 双基础 ISO、优秀弱光表现 | 夜景、暗光环境 |
| Super 8mm / 16mm Film | 高颗粒、温暖色调、有机质感 | 怀旧、纪录片、Vlog |

### 镜头焦段参考

| 焦段 | 特征 | 适用 |
|---|---|---|
| 16mm Ultra-wide | 极端透视畸变、环境包裹感 | 建立镜头、空间展示 |
| 24mm Wide | 轻微畸变、环境+主体平衡 | 场景建立、群戏 |
| 35mm Standard-wide | 自然人眼视角、叙事万能焦段 | 叙事主体、行走跟随 |
| 50mm Normal | 最接近人眼、无畸变 | 对话、日常 |
| 85mm Portrait | 压缩背景、柔美虚化 | 人物特写、情感表达 |
| 135mm Telephoto | 强烈压缩、极浅景深 | 远距离观察、孤立感 |

---

## 物理缺陷法 (Physical Imperfections)

完美无瑕的主体看起来像假的。在 prompt 中加入磨损、瑕疵、老化痕迹能显著提升真实感：

### 人物层面
- battle scars, weathered skin, calloused hands
- worn clothing with tears and patches
- dusty boots, scuffed armor, tarnished jewelry

### 环境层面
- peeling paint, cracked concrete, rust streaks
- broken windows, overgrown vegetation reclaiming structures
- water stains, faded signage, flickering lights

### 物体层面
- scratches and dents on metal surfaces
- fingerprints on glass, dust accumulation
- frayed cables, exposed wiring, mismatched repair patches

---

## 静默结尾法 (Quiet Conclusions)

避免爆炸式高潮，选择悬而未决的视觉时刻：

- No dialogue, no light burst — just rain vaporizing on the shield
- The camera lingers on an empty doorway after the character exits
- Final frame: a single object left behind, slowly coming into focus
- The last shot holds on a landscape with one subtle moving element

---

## AI 视频生成模型特性速查

| 模型 | 强项 | 最佳 Prompt 风格 |
|---|---|---|
| Sora 2 | 物理模拟、长镜头、复杂场景 | 详细叙事描述、自然语言 |
| Kling 3.0+ | 人物一致性、面部表情 | 结构化标签 + 叙事混合 |
| Wan2.7 | 图生视频、运镜控制 | 简洁运镜指令 + 首帧引导 |
| Veo 3.1 | 电影质感、光影细节 | 电影设备规格 + 光影描述 |
| Runway Gen-4 | 风格一致性、转场 | 视觉风格锚定 + 转场指令 |

---

## 常见负面 Prompt 模式（应避免）

| 避免 | 替代方案 |
|---|---|
| "highly detailed, 4K, photorealistic" (泛化堆砌) | 指定具体设备: "shot on ARRI Alexa, 35mm anamorphic" |
| "epic cinematic shot" (空洞修饰) | 描述具体运镜: "slow crane ascending from medium to epic wide" |
| "dramatic lighting" (不明确) | 指定光照方案: "rim-lit from behind with warm amber, key light from upper-left" |
| "beautiful" / "stunning" (主观形容词) | 描述具体视觉特征: "golden hour warmth on weathered skin, shallow DOF bokeh" |
| 过多主体和动作 (超出模型能力) | 每个镜头聚焦一个核心主体和一个核心动作 |
| 矛盾的灯光描述 | 确保主光、辅光、轮廓光方向逻辑一致 |

---

## Negative Prompt 参考（SD/MJ 系）

```
worst quality, low quality, blurry, deformed, disfigured, bad anatomy, bad hands,
extra fingers, missing fingers, extra limbs, fused fingers, text, watermark, signature,
cropped, out of frame, duplicate, ugly, mutation, morbid, mutilated
```

对于视频生成额外添加：
```
flickering, morphing face, inconsistent lighting between frames, jittery camera,
temporal inconsistency, warping background, melting objects
```
