from pathlib import Path
import json, shutil

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "output" / "keyframes" / "_legacy"
LIB_ROOT = ROOT / "output" / "reference_sheets" / "library"

# 用户已命名的素材映射：filename -> (category, name, label, description)
USER_ASSETS = {
    "user_仙尊_1.jpg": ("characters", "char_elder_master_front", "师尊·正面", "白发白须仙人师尊正面全身，手持玉如意，云海仙山背景"),
    "user_仙尊.jpg":    ("characters", "char_elder_master_side", "师尊·侧面", "白发仙人师尊侧面像，衣袂飘飘，仙山楼阁背景"),
    "user_女剑仙_1.jpg":("characters", "char_female_companion_half", "女剑仙·半身", "青衣白发女剑仙半身像，晚霞仙山背景"),
    "user_女剑仙.jpg":  ("characters", "char_female_companion_full", "女剑仙·全身", "青衣白发女剑仙全身立绘，云海山巅"),
    "user_青锋剑.png":  ("props", "prop_qingfeng_sword", "青锋剑", "环绕星宿符文阵法的青色飞剑，流苏剑穗"),
    "user_神秘玉佩.png":("props", "prop_mystic_jade_pendant", "神秘玉佩", "白玉貔貅雕纹玉佩，红绳中国结，黑烟背景"),
    "user_魔尊_1.jpg":  ("characters", "char_villain_lord_front", "魔尊·正面", "骷髅面具反派首领正面全身，红黑长袍，血月背景"),
    "user_魔尊.jpg":    ("characters", "char_villain_lord_side", "魔尊·侧面", "魔尊侧面像，持魔剑，血月与碎石背景"),
}

def main():
    manifest_path = LIB_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = {m["name"]: m for m in manifest}

    for fname, (category, name, label, description) in USER_ASSETS.items():
        src = LEGACY / fname
        if not src.exists():
            print(f"[SKIP] {fname} not found")
            continue
        dst_dir = LIB_ROOT / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{name}{src.suffix.lower()}"
        if dst.exists():
            dst = dst_dir / f"{name}_user{src.suffix.lower()}"
            print(f"[INFO] conflict resolved, copying to {dst}")
            entry["name"] = f"{name}_user"
            entry["url"] = f"/media/reference_sheets/library/{category}/{dst.name}"
            entry["filename"] = dst.name
        shutil.copy2(src, dst)
        print(f"[COPY] {fname} -> {dst}")

        entry = {
            "filename": dst.name,
            "category": category,
            "name": name,
            "label": label,
            "description": description,
            "source": "user",
            "url": f"/media/reference_sheets/library/{category}/{dst.name}",
        }
        if name in existing:
            existing[name] = entry
        else:
            manifest.append(entry)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] manifest updated（{len(manifest)} 条）")


if __name__ == "__main__":
    main()
