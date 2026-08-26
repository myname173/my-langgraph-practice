"""整理素材库：把 output/keyframes/_legacy 下的用户真实参考图（user_*.jpg/png）规范命名后
移入 output/reference_sheets/library，并写入 manifest.json；然后清理 _legacy 目录。

注意：jimeng 生成的关键帧已在 output/keyframes/_generated/，绝不进参考图库。
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "output" / "keyframes" / "_legacy"
LIB = ROOT / "output" / "reference_sheets" / "library"

# 源文件(在 _legacy) -> (目标子目录, 规范文件名, manifest 条目)
# 规范命名遵循 library 现有约定：char_<role>_<pose>.jpg / prop_<name>.png
MAP = {
    "user_仙尊.jpg": ("characters", "char_elder_master_real.jpg",
        {"label": "仙尊·真人参考", "role_name": "仙尊",
         "description": "用户提供的真人参考图：仙尊，雪白道袍银线云鹤纹，银发束冠玉簪，手持青玉如意"}),
    "user_仙尊_1.jpg": ("characters", "char_elder_master_real_alt.jpg",
        {"label": "仙尊·真人参考(侧面)", "role_name": "仙尊",
         "description": "用户提供的真人参考图：仙尊侧面指路姿态，白袍银线云鹤纹，身後瀑布桃花云海"}),
    "user_魔尊.jpg": ("characters", "char_villain_lord_real.jpg",
        {"label": "魔尊·真人参考", "role_name": "魔尊",
         "description": "用户提供的真人参考图：魔尊，玄黑鎏金龙纹长袍，骷髅骨面半盔，红月背景"}),
    "user_魔尊_1.jpg": ("characters", "char_villain_lord_real_alt.jpg",
        {"label": "魔尊·真人参考(正面)", "role_name": "魔尊",
         "description": "用户提供的真人参考图：魔尊正面全身，半面骷髅面具，双手握剑竖立胸前"}),
    "user_女剑仙.jpg": ("characters", "char_female_companion_real.jpg",
        {"label": "女剑仙·真人参考", "role_name": "女剑仙",
         "description": "用户提供的真人参考图：女剑仙，青碧色长袍云纹袖口，翡翠腰带，长发半束"}),
    "user_女剑仙_1.jpg": ("characters", "char_female_companion_real_alt.jpg",
        {"label": "女剑仙·真人参考(侧面持剑)", "role_name": "女剑仙",
         "description": "用户提供的真人参考图：女剑仙侧面持剑，青碧长袍同款，暮色山峦背景"}),
    "user_青锋剑.png": ("props", "prop_qingfeng_sword_real.png",
        {"label": "青锋剑·真人参考", "role_name": "青锋剑",
         "description": "用户提供的本命青锋飞剑参考图，玉蓝透光剑身云纹，银柄流苏"}),
    "user_神秘玉佩.png": ("props", "prop_mystic_jade_pendant_real.png",
        {"label": "神秘玉佩·真人参考", "role_name": "神秘玉佩",
         "description": "用户提供的古玉佩参考图，羊脂白玉雕仙兽图腾，金芒内蕴，红丝绦"}),
}

# 读取 _legacy 原 manifest 以复用描述（若有更详细的）
legacy_manifest = {}
lm_path = LEGACY / "manifest.json"
if lm_path.exists():
    try:
        for e in json.loads(lm_path.read_text(encoding="utf-8")):
            legacy_manifest[e["filename"]] = e
    except Exception:
        pass

# 载入现有 library manifest
manifest_path = LIB / "manifest.json"
existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
existing_names = {e["filename"] for e in existing}

added = 0
for src_name, (sub, dst_name, meta) in MAP.items():
    src = LEGACY / src_name
    if not src.exists():
        print(f"  [skip] 源文件不存在: {src_name}")
        continue
    dst_dir = LIB / sub
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / dst_name
    shutil.move(str(src), str(dst))
    print(f"  [move] {src_name} -> {sub}/{dst_name}")
    # 若已存在同名条目则跳过
    if dst_name in existing_names:
        print(f"    [note] manifest 已有 {dst_name}，跳过追加")
        added += 1
        continue
    # 优先用 _legacy manifest 的详细描述，否则用 MAP 里的
    lm = legacy_manifest.get(src_name, {})
    entry = {
        "filename": dst_name,
        "category": sub,
        "name": dst_name.rsplit(".", 1)[0],
        "label": meta["label"],
        "role_name": meta["role_name"],
        "description": lm.get("description") or meta["description"],
        "source": "user",
        "url": f"/media/reference_sheets/library/{sub}/{dst_name}",
    }
    existing.append(entry)
    existing_names.add(dst_name)
    added += 1

# 写回 manifest
manifest_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[OK] manifest 更新完成，共 {len(existing)} 条，本次新增/保留 {added} 条")

# 清理 _legacy：移空后删除（关键帧已在 _generated，_legacy 仅作旧 dumping ground）
remaining = [p for p in LEGACY.iterdir()] if LEGACY.exists() else []
if remaining:
    print(f"  [warn] _legacy 仍有残留文件未处理: {[p.name for p in remaining]}")
else:
    shutil.rmtree(LEGACY, ignore_errors=True)
    print("  [OK] 已清理空的 _legacy 目录")
