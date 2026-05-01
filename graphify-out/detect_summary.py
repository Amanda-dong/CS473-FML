import json
from graphify.detect import detect
from pathlib import Path
r = detect(Path("."))
print(f"total_files={r['total_files']} total_words={r['total_words']} warning={r.get('warning')}")
code = r['files'].get('code', [])
docs = r['files'].get('document', [])
print(f"code={len(code)} docs={len(docs)}")
