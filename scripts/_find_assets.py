import os, glob
root = "c:/Users/13682/Desktop/my-langgraph-practice-main"
for name in ["char_main.jpg", "char_villain1.jpg", "char_villain2.jpg"]:
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    print(name, "->", hits if hits else "NOT FOUND")
# also list assets dir
assets_dir = os.path.join(root, "assets")
if os.path.isdir(assets_dir):
    print("=== assets/ ===")
    for f in os.listdir(assets_dir):
        print(" ", f)
