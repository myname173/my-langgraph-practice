import urllib.request, json
# 探测 jimeng 是否真有免费额度：发一个极小的文生图请求测试（不保存到正式产物，纯探活/额度）
body = {
    "model": "jimeng-3.0",
    "prompt": "test probe, a red dot",
    "n": 1,
    "size": "256x256"
}
req = urllib.request.Request("http://localhost:8090/v1/images/generations",
    data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
        print("jimeng ok, keys:", list(d.keys()))
        print(json.dumps(d, ensure_ascii=False)[:300])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read(300).decode(errors="ignore"))
except Exception as e:
    print("ERR", str(e)[:200])
