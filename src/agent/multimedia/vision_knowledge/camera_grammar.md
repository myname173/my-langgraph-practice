# Camera Grammar — Cinematography Knowledge Base

Structured camera movement vocabulary for AI-generated game trailers.
Each entry maps a shot type to its narrative function and emotional effect.

---

## Shot Scale

| Shot Type | Narrative Function | Emotional Effect |
|---|---|---|
| extreme wide shot | establish vast environment, show scale of world | awe, isolation, epic scope |
| wide shot | establish world scale, show character in context | context-setting, grandeur |
| medium shot | balance character and environment | conversational, grounded |
| close-up | focus on character emotion or key object | intimacy, tension, emphasis |
| extreme close-up | reveal fine detail (eye, hand, weapon) | suspense, dramatic focus |

## Camera Angle

| Angle | Narrative Function | Emotional Effect |
|---|---|---|
| eye level | neutral observation | relatability, realism |
| low angle | look up at subject | power, dominance, heroism |
| high angle | look down at subject | vulnerability, insignificance |
| bird's eye | top-down overhead view | god-like perspective, strategy |
| dutch angle | tilted horizon | unease, disorientation, chaos |
| over-the-shoulder | POV adjacency | confrontation, connection |

## Camera Movement

| Movement | Narrative Function | Emotional Effect |
|---|---|---|
| dolly in | move camera toward subject | emotional escalation, focus narrowing |
| dolly out | move camera away from subject | emotional withdrawal, reveal context |
| tracking shot | camera follows subject laterally | action continuity, journey |
| orbit shot | camera circles around subject | character reveal, heroic moment |
| pan (horizontal) | rotate camera on fixed axis | survey scene, reveal information |
| tilt (vertical) | rotate camera up/down on fixed axis | scale reveal, power dynamic |
| crane up | elevate camera vertically | transcendence, overview, triumph |
| crane down | descend camera toward ground | grounding, intimacy, descent |
| steadicam follow | smooth following movement | immersion, first-person feel |
| handheld | unstable, organic movement | urgency, realism, chaos |
| whip pan | rapid horizontal rotation | surprise, time skip, energy burst |
| zoom in (lens) | magnify without moving camera | sudden focus, shock, realization |
| zoom out (lens) | widen without moving camera | context reveal, loneliness |

## Movement Combinations

| Combination | Use Case | Example |
|---|---|---|
| dolly in + low angle | hero power-up moment | character draws weapon, camera pushes in from below |
| orbit + crane up | character reveal / transformation | full rotation while ascending to show new form |
| tracking + handheld | chase or combat sequence | following character through environment with urgency |
| pan + tilt | environment survey | sweeping across landscape from ground to sky |
| dolly out + crane up | farewell / departure | pulling away and rising as character walks into distance |

## Prompt Keywords for Video Generation

When writing camera movement prompts for AI video generation models, use these precise descriptors:

- `Slow dolly in` — gradual push, 3-5 seconds
- `Smooth tracking left/right` — lateral follow
- `360 orbit around subject` — full rotation
- `Low angle tilt up` — ground-to-sky reveal
- `Steady handheld follow` — organic chase feel
- `Static locked tripod` — no movement, stability
- `Cinematic crane ascending` — vertical rise
- `Whip pan transition` — rapid directional cut
