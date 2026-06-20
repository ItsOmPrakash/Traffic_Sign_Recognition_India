import json

with open("data/raw/ITS_Night_dataset1/annotations.json") as f:
    data = json.load(f)

for cat in data["categories"]:
    print(cat["id"], "->", cat["name"])