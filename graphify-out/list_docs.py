import json
d = json.load(open("graphify-out/.graphify_detect.json"))
docs = d["files"].get("document", [])
code = d["files"].get("code", [])
print("docs:", docs)
print("code_count:", len(code))
