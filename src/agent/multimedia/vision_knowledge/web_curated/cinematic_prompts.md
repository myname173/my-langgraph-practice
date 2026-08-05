# 电影级 AI 视频 / 图像 Prompt 精选

> 来源：GitHub ai-shortfilm-prompts、Microsoft Learn Sora Case Studies、Sora 社区实战、Midjourney/SD 高赞作品。
> 用途：游戏宣传片、AI 短剧、电影级视觉内容的 prompt 参考。

---

## 5 阶段高质量 Prompt 框架

来自 GitHub ai-shortfilm-prompts (MIT License) 的方法论：

1. **Core Theme** — 用管道分隔的风格标签定义核心主题
2. **Character & Environment** — 角色和环境的具体描述
3. **Atmosphere & Color** — 大气效果和调色参数
4. **Camera Movement** — 运镜方式和角度规格
5. **Temporal Storyboard** — 按秒/按镜头的时间轴编排

### 通用基底模板 (Universal Base Template)

```
Anamorphic widescreen cinematic. Simulated IMAX film camera +
Panavision C-series lens (35mm focal, f/4 aperture). Handheld
shot — extremely subtle, breath-like camera float throughout.
{{scene description}}.
No score. Production audio only.
```

### 有效 vs 无效的 Prompt 对比

**无效方式（泛化修饰词堆砌）：**
```
Epic cinematic shot of a beautiful female mech warrior activating a
stunning energy shield in the rain. Highly detailed, 4K, photorealistic,
movie-quality, dramatic lighting.
```

**有效方式（5 阶段框架）：**
```
Core theme: gritty hard sci-fi mech | rainy dock | battle-damage aesthetic | energy shield | post-apocalyptic live-action
Atmosphere: simulated IMAX film camera + Panavision C-series (35mm, f/4). Low-saturation teal base, film grain.
Camera: handheld — extremely subtle, breath-like float throughout.
9-12s: hexagonal energy cells light up unevenly, some flicker as if faulty; rain bends around a 2m dome.
Ending: no dialogue, no light burst — just rain vaporizing on the shield, a lightning flash across the dock.
```

---

## 赛博朋克都市雨夜

**场景类型：** environment / cyberpunk
**适用模型：** Sora / Kling / Wan2.7

```
Core theme: cyberpunk metropolis | rain-soaked streets | neon reflections | solitary figure
Atmosphere: simulated ARRI Alexa 65 + anamorphic lens (40mm, f/2.8). Teal-orange color grading, heavy film grain, volumetric fog.
Camera: slow dolly-in from wide to medium, subtle handheld drift.
Subject: A fashionable woman walks through neon-lit, warm-humid Tokyo streets lined with dense Japanese signage and warm lanterns.
Lighting: neon cold light mixed with warm amber practicals, blue-purple contrast lighting, strong chiaroscuro.
Details: rain puddles reflecting holographic advertisements, steam rising from street grates, distant skyscraper silhouettes in fog.
```

---

## 超现实鲸鱼都市穿越

**场景类型：** environment / surreal
**适用模型：** Sora / Veo

```
Core theme: surreal dreamscape | giant translucent whale | modern metropolis | golden hour
Atmosphere: simulated IMAX 70mm film. Warm golden color palette, lens flare, atmospheric haze.
Camera: slow crane ascending from street level to rooftop height.
Subject: A massive, translucent whale glides gracefully between modern skyscrapers at sunset.
Lighting: golden hour side-lighting, building glass reflecting warm amber light, volumetric god rays through whale body.
Details: the whale's bioluminescent patterns pulse gently, pedestrians below freeze in awe, birds scatter from building ledges.
```

---

## 微距纪录片：树叶纸船竞赛

**场景类型：** macro / nature documentary
**适用模型：** Kling / Wan2.7

```
Core theme: macro nature documentary | miniature boats | rain stream | moss and pebbles
Atmosphere: simulated macro lens (100mm, f/2.8). Shallow depth of field, natural daylight, water droplet detail.
Camera: low-angle tracking shot following the stream, slight handheld bobbing.
Subject: A fleet of tiny paper boats made from leaves and bark races along a rainwater stream flowing over pebbles and moss.
Lighting: soft diffused natural light filtering through canopy, specular highlights on water surface.
Details: each boat navigates differently around obstacles, water splashes in slow motion, tiny insects watch from mossy banks.
```

---

## 机甲战士能量盾（硬科幻）

**场景类型：** action / sci-fi mech
**适用模型：** Sora / Kling

```
Core theme: gritty hard sci-fi mech | rainy industrial dock | battle-damage aesthetic | energy shield activation
Atmosphere: simulated IMAX film camera + Panavision C-series (35mm, f/4). Low-saturation teal base, heavy film grain.
Camera: handheld — extremely subtle breath-like float throughout.
Subject: A battle-scarred mech warrior (3m tall, exposed hydraulics, scorch marks on left arm) activates a hexagonal energy shield in pouring rain.
9-12s: hexagonal energy cells light up unevenly, some flicker as if faulty; rain bends around a 2m dome creating steam curtains.
Lighting: industrial sodium vapor orange from dock lights, cold blue from energy shield, high-contrast chiaroscuro.
Ending: no dialogue, no light burst — just rain vaporizing on the shield surface, a distant lightning flash illuminating the entire dock.
```

---

## 浮空城市音乐视觉

**场景类型：** environment / sci-fi
**适用模型：** Sora / Veo

```
Core theme: futuristic floating cityscape | nighttime | pulsing lights | solitary figure on rooftop
Atmosphere: simulated wide anamorphic lens. Deep blue-black palette with warm accent lights.
Camera: slow orbital track around the rooftop figure, gradually revealing the city below.
Subject: A character stands on a rooftop overlooking a glowing futuristic skyline at night.
Lighting: city lights pulsing in rhythmic patterns, warm amber from the character's suit, cool cyan from holographic billboards.
Details: layered city environment with depth, flying vehicles leaving light trails, slow fog drift between building clusters.
```

---

## 迷雾森林奇幻旅人

**场景类型：** environment / fantasy
**适用模型：** Kling / Wan2.7

```
Core theme: enchanted misty forest | bioluminescent mushrooms | shifting shadows | lone traveler
Atmosphere: simulated 35mm film. Desaturated green-teal base, volumetric fog, practical light sources.
Camera: Steadicam follow from behind, slowly pushing past the traveler into the forest depth.
Subject: A hooded traveler walks through a mist-filled forest with glowing mushrooms and moving shadows.
Lighting: bioluminescent blue-green from mushroom clusters, dappled moonlight through canopy, warm torch glow from traveler's hand.
Details: atmospheric fog swirls with each footstep, small forest creatures' eyes glow briefly in the undergrowth, ancient tree roots form archways.
```

---

## 深空太阳系全景

**场景类型：** environment / space
**适用模型：** Sora / Veo

```
Core theme: cinematic solar system | deep space | orbital motion | luminous trails
Atmosphere: simulated IMAX 70mm. Deep black background, high-contrast planetary lighting, lens flare from sun.
Camera: rotating orbital camera with slow zoom-out, scale transitions from close planet surface to full system view.
Subject: Planets orbit the sun in a cinematic deep-space view with rotating camera and glowing orbital trails.
Lighting: harsh directional sunlight creating terminator lines on planets, reflected light from gas giant atmospheres.
Details: asteroid belt particles catch sunlight, a comet trail crosses the frame, layered depth with parallax between near and far planets.
```

---

## 国风水墨意境大片

**场景类型：** environment / Chinese ink painting
**适用模型：** Kling / Wan2.7

```
Core theme: traditional Chinese ink painting | ethereal mountain landscape | flowing water | poetic atmosphere
Atmosphere: sumi-e ink wash style. Monochrome with subtle indigo accents, visible brush strokes, rice paper texture.
Camera: slow crane pull-out from close detail to vast landscape, overhead angle gradually tilting.
Subject: Mist-shrouded mountain peaks with cascading waterfalls, a lone fisherman on a bamboo raft, pine trees clinging to cliff faces.
Lighting: diffused ambient light, no harsh shadows, ink-wash gradient from dark foreground to light background.
Constraint: maintain consistent ink painting aesthetic throughout, no modern elements, no photorealistic textures.
```

---

## 产品宣传大片风格

**场景类型：** product / commercial
**适用模型：** Kling / Wan2.7

```
Core theme: luxury product showcase | minimalist aesthetic | studio lighting | 360 rotation
Atmosphere: simulated medium format camera (80mm, f/5.6). Clean neutral background, controlled studio environment.
Camera: slow 360-degree orbit around the product, maintaining eye-level perspective.
Subject: [product] slowly rotating to reveal every surface detail, material texture, and craftsmanship.
Lighting: three-point studio setup — key light from upper-left (soft diffused), fill from right, rim light from behind creating edge separation.
Constraint: no background clutter, no watermark, perfectly stable frame, subject always in sharp focus.
```

---

## 复古胶片 Vlog 风格

**场景类型：** lifestyle / nostalgic
**适用模型：** Kling / Wan2.7

```
Core theme: retro film aesthetic | nostalgic mood | warm tones | personal documentary
Atmosphere: simulated Super 8mm film. Warm amber color shift, visible film grain, slight vignetting, occasional light leak.
Camera: handheld with organic movement, occasional lens whacking for flare effects.
Subject: everyday moments captured with intimacy — morning coffee steam, afternoon sunlight through curtains, evening street walk.
Lighting: natural available light only, golden hour warmth, window light creating soft shadows.
Details: slight film gate weave, subtle focus breathing, organic lens characteristics.
```

---

## Midjourney / SD 高质量 Prompt 结构

来自 Stable Diffusion 社区的通用高质量 prompt 5 层结构：

### Layer 1: Subject (主体描述)
```
A majestic volcano erupting under a starry night sky, with lava flowing into the sea of sand.
```

### Layer 2: Detail Modifiers (细节修饰)
```
cinematic scene, dramatic lighting highlighting the lava, volumetric smoke, highly detailed, sharp focus
```

### Layer 3: Color Tones (色彩基调)
```
Gothic gloomy atmosphere, washed color tones, triadic color scheme of deep red, midnight blue, and ash gray
```

### Layer 4: Art Style (风格锚定)
```
concept art style, digital matte painting, trending on ArtStation
```

### Layer 5: Renderer (渲染引擎)
```
rendered in Unreal Engine 5, Octane Render, 8K resolution, ray-traced global illumination
```

### 组合示例：赛博朋克暗巷

```
Futuristic cyberpunk slum alleyway, heavy rain, neon lights reflecting on wet pavement, intricate cables and pipes, towering background skyscrapers, cinematic lighting, volumetric fog, photorealistic, Unreal Engine 5 rendering, 8K, extreme detail, 16:9 --ar 16:9 --v 6.0
```

### 组合示例：奇幻浮空遗迹

```
Wide angle shot of ancient floating ruins in the sky, covered in moss and vines, cascading waterfalls, fantasy world, majestic clouds, golden hour sunlight, god rays, concept art style blending Studio Ghibli and Elden Ring, digital painting, ArtStation trending, 21:9 --ar 21:9 --s 250
```

### 组合示例：等距废土基地

```
Isometric view of a post-apocalyptic survivor camp, scrap metal walls, campfire in the center, desert wasteland, loot crates and supply barrels, diorama style, 3D rendering via Blender, sharp focus, white background for UI extraction, 1:1 --ar 1:1
```

---

## Prompt 权重分配建议

基于社区实战总结的 5 层权重结构（适用于剧本式视频 prompt）：

| 层级 | 权重 | 说明 |
|---|---|---|
| 角色设定 | 15% | 人物外观、服装、表情、状态 |
| 场景环境 | 15% | 地理位置、建筑风格、天气、时段 |
| 动作序列 | 40% | 核心行为、连续动作、物理交互（最重要的部分） |
| 对话交互 | 15% | 语言、说话方式、情绪副词 |
| 风格标签 | 15% | 画面风格、调色、渲染质量、镜头语言 |

**动作描写逻辑：** 基础 → 细节 → 连续 → 结果（递进式）

**对话与情绪控制：** 涉及人物交互时必须指定语种，标注说话动词和情绪副词（如 "shouts firmly", "whispers anxiously"）。
