import sys

open("/home/bossman/projects/cs473-fml/graphify-out/.graphify_python", "w").write(
    sys.executable
)
print("saved:", sys.executable)
