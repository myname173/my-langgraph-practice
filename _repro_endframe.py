import os, sys, glob
sys.path.insert(0, r'c:\Users\13682\Desktop\my-langgraph-practice-main\.venv\Lib\site-packages')
envf = r'c:\Users\13682\Desktop\my-langgraph-practice-main\.env'
for line in open(envf, encoding='utf-8'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not os.environ.get(k):
            os.environ[k] = v

from src.agent.multimedia.tools import image_gen, jimeng

# 用首帧实际生成的 PNG
kf = r'c:\Users\13682\Desktop\my-langgraph-practice-main\output\keyframes\_generated\jimeng_kf_1787600605965.png'
print("kf exists:", os.path.exists(kf), "size:", os.path.getsize(kf) if os.path.exists(kf) else 0)

# 检查 _as_inline 对该绝对路径的输出
inline = jimeng._as_inline(kf)
print("inline type:", inline[:30], "len:", len(inline))

neg = ("same pose as reference, identical composition, static, frozen, no movement, "
       "copy of reference image, unchanged position, stiff body, no expression change, duplicate")
prompt = "a heroic cinematic shot of a sword cultivator, final frame, different dynamic stance\nAvoid: " + neg

# 完整复现尾帧调用
try:
    r = image_gen.generate_keyframe(prompt, reference_image_urls=[kf], ref_strength=0.08, negative_prompt=neg, thread_id="repro")
    print("END_FRAME OK ->", r)
except Exception as e:
    print("END_FRAME FAIL ->", repr(e)[:200])
