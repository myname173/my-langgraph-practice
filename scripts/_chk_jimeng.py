import urllib.request, json, sys

url = "http://localhost:8090/v1/models"
try:
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))
    print("JIMENG_API_OK")
    print(json.dumps(data)[:300])
except Exception as e:
    print("JIMENG_API_ERR:", repr(e))
    sys.exit(1)
