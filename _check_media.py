import urllib.request
try:
    req = urllib.request.Request('http://localhost:8900/', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=4) as r:
        print('MEDIA_UP', r.status)
except Exception as e:
    print('MEDIA_DOWN', repr(e)[:120])

# 确认 graph anchor url 形式：grep _match_reference_elements 返回
import re
g = open(r'c:\Users\13682\Desktop\my-langgraph-practice-main\src\agent\multimedia\graph.py', encoding='utf-8').read()
# 找 anchor_urls 赋值来源
for m in re.finditer(r'anchor_urls\s*=\s*(.+)', g):
    print('ANCHOR_ASSIGN:', m.group(1)[:100])
# ref_match 来源
for m in re.finditer(r'_match_reference_elements\(([^)]*)\)', g):
    print('MATCH_CALL:', m.group(1)[:120])
