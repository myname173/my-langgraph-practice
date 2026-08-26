from pathlib import Path
from PIL import Image
import imagehash, json, sys

root = Path("output/keyframes/_legacy")
files = sorted(root.rglob("*"))
imgs = [p for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
print(f"total keyframe images: {len(imgs)}", file=sys.stderr)

hashes = []
for p in imgs:
    try:
        h = imagehash.phash(Image.open(p))
        hashes.append((p, h))
    except Exception as e:
        print(f"skip {p}: {e}", file=sys.stderr)

clusters = []
for p, h in hashes:
    placed = False
    for c in clusters:
        if h - c["hash"] <= 8:
            c["items"].append(p)
            placed = True
            break
    if not placed:
        clusters.append({"hash": h, "items": [p]})

clusters.sort(key=lambda c: len(c["items"]), reverse=True)
print(f"clusters: {len(clusters)}", file=sys.stderr)

report = []
for i, c in enumerate(clusters):
    rep = c["items"][0]
    report.append({
        "group": i + 1,
        "size": len(c["items"]),
        "representative": str(rep),
        "hash": str(c["hash"]),
        "all": [str(p) for p in c["items"][:4]]
    })
    print(f"[{i+1}] size={len(c['items']):3d} rep={rep} hash={c['hash']}")

with open("output/keyframes/_legacy/_kf_clusters.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
