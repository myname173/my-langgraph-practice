from pathlib import Path
import json
root = Path('output/keyframes/_generated')
files = sorted([p for p in root.iterdir() if p.suffix.lower()=='.png'])
manifest = []
for f in files:
    ts = f.stem.replace('jimeng_kf_','')
    manifest.append({
        'filename': f.name,
        'category': 'keyframe',
        'name': f.stem,
        'label': f'即梦关键帧 {ts}',
        'description': '由 jimeng 生成的关键帧/尾帧历史产物，不归入参考图素材库',
        'source': 'keyframe',
        'url': f'/media/keyframes/_generated/{f.name}'
    })
(root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'[OK] {len(manifest)} keyframes manifest generated')
