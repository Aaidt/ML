import os
import json
from pathlib import Path

def read_file(filename: str) -> str:
    exists = os.path.exists(filename)
    is_file = Path.is_file(filename)

    if exists and is_file:
        with open(filename, "r") as f:
            content = f.read()

        return json.dumps({"path": filename, "content": content[:3000], "truncated": len(content) > 3000})
    else:     
        return json.dumps({"error": "invalid file path"})


def write_file(filename: str, code: str) -> None:
    try:
        d = os.path.dirname()
        if d:
            os.makedirs(d, exist_ok=True)
        with open(filename, "w") as f:
            f.write(code)
            return json.dumps({"status": "success", "path": filename, "bytes": len(code)})
    except Exception as e:     
        return json.dumps({"error": "invalid file path"})


# 
# def run_code(path: str) -> None:
# 
# 
# def list_dirs(path: str) -> str:
