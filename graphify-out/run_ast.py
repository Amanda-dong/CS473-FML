import json, sys
from graphify.extract import collect_files, extract
from graphify.detect import detect
from pathlib import Path

r = detect(Path("."))
with open("graphify-out/.graphify_detect.json", "w") as f:
    json.dump(r, f, indent=2)

code_files = []
for fp in r.get("files", {}).get("code", []):
    p = Path(fp)
    if p.is_dir():
        code_files.extend(collect_files(p))
    else:
        code_files.append(p)

if code_files:
    result = extract(code_files, cache_root=Path("."))
    with open("graphify-out/.graphify_ast.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"AST: {len(result['nodes'])} nodes, {len(result['edges'])} edges")
else:
    with open("graphify-out/.graphify_ast.json", "w") as f:
        json.dump({"nodes": [], "edges": [], "input_tokens": 0, "output_tokens": 0}, f)
    print("No code files")
