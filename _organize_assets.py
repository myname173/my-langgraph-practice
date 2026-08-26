from pathlib import Path
import json, shutil, os

ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT / "output" / "assets_uploaded"
LIB_ROOT = ROOT / "output" / "reference_sheets" / "library"

# 已人工识图后的分类映射：group -> (category, name, label, description)
# 基于 output/_asset_clusters.json 的 group 编号
GROUP_MAP = {
    1:  ("characters", "char_villain1", "反派一·红黑长袍", "反派一号角色三视图设定，红黑长袍、双剑、凌厉神态"),
    2:  ("characters", "char_villain2", "反派二·伤痕锁链", "反派二号角色三视图设定，黑衣破损、锁链缠身、阴郁神态"),
    3:  ("characters", "char_main", "主角·白衣青剑", "主角角色三视图设定，白衣长衫、青玉剑、仙侠少年"),
    4:  ("props", "prop_bloodied_blade", "血染环首刀", "带血环首刀武器，红缨刀穗，岩石背景"),
    5:  ("props", "prop_demon_sword_bone", "魔骨剑", "反派骨剑，缠绕锁链与血红符文"),
    6:  ("props", "prop_jade_sword_scabbard", "青玉剑·剑鞘", "主角青玉剑与剑鞘，玉佩流苏"),
    7:  ("environments", "env_floating_celestial_palace", "悬浮天宫", "云海之上悬浮的仙山宫殿、亭台楼阁"),
    8:  ("environments", "env_ancient_tower_storm", "雷暴镇妖塔", "雷电交加的镇妖塔场景，乌云密布"),
    9:  ("other", "char_modern_portrait_male", "现代证件照·男", "白衬衫黑领带的现代男性证件照，非仙侠素材"),
    11: ("environments", "env_megastructure_peak", "巨构山峰", "未来感巨构建筑与山峰远景"),
    12: ("props", "prop_megastructure_array", "巨构阵列", "未来感环形阵列/空间站"),
    13: ("props", "prop_jade_pendant_kylinc", "白玉貔貅玉佩", "白玉貔貅雕纹玉佩，红绳"),
    14: ("props", "prop_flying_sword_stars", "星宿飞剑", "浮于星宿阵法中的飞剑，符文光芒"),
    15: ("props", "prop_jade_pendant_dragon_bell", "龙纹玉佩·青铜铃", "龙纹玉佩、青铜铃与云海仙山"),
    16: ("props", "prop_flying_sword_clouds", "云海仙剑", "悬浮于云海之上的仙剑，带符文光芒"),
    17: ("environments", "env_classical_mountain_gate", "水墨山门", "水墨山水风的仙门/宗门入口"),
    18: ("environments", "env_demonic_ruins_blood", "血色禁地", "破碎石碑与血月笼罩的废墟场景"),
    19: ("environments", "env_spirit_cave_treasure", "洞天福地", "悬浮宝箱与灵符环绕的秘境洞府"),
    20: ("characters", "char_female_companion_half", "师姐·半身", "青衣白发女性角色半身像，仙侠师姐"),
    21: ("characters", "char_female_companion_full", "师姐·全身", "青衣白发女性角色全身立绘"),
    22: ("characters", "char_elder_master_half", "师尊·半身", "白发白须老者/仙人师尊半身像"),
    23: ("characters", "char_villain_lord_full", "魔尊·全身", "反派首领全身像，骷髅面具、红月背景"),
    24: ("characters", "char_villain_lord_profile", "魔尊·侧面", "反派首领侧面像"),
    25: ("characters", "char_elder_master_full", "师尊·全身", "白发白须仙人师尊全身像"),
}


def choose_best(paths: list[str]) -> Path:
    """从同组文件里选最大、最存在的文件作为规范版本。"""
    candidates = [Path(p) for p in paths if Path(p).exists()]
    if not candidates:
        raise FileNotFoundError(f"No valid file in {paths[:3]}...")
    # 优先 png，其次按文件大小
    return max(candidates, key=lambda p: (p.suffix.lower() == ".png", p.stat().st_size))


def main():
    report = json.loads((ROOT / "output" / "_asset_clusters.json").read_text(encoding="utf-8"))
    manifest = []
    for item in report:
        g = item["group"]
        if g not in GROUP_MAP:
            print(f"[SKIP] group {g} 无映射")
            continue
        category, name, label, description = GROUP_MAP[g]
        if category == "other":
            # 现代证件照与仙侠无关，不进入参考图库
            print(f"[SKIP] group {g} 为 other，不入库: {label}")
            continue
        src = choose_best(item["all"])
        dst_dir = LIB_ROOT / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{name}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        print(f"[COPY] {src.name} -> {dst}")
        manifest.append({
            "filename": dst.name,
            "category": category,
            "name": name,
            "label": label,
            "description": description,
            "source": "user",
            "url": f"/media/reference_sheets/library/{category}/{dst.name}",
        })

    # 写 manifest
    manifest_path = LIB_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] manifest.json -> {manifest_path}（{len(manifest)} 条）")


if __name__ == "__main__":
    main()
