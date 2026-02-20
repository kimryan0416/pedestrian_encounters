from pathlib import Path
from datetime import datetime

def find_recording_dirs(
    root: Path,
    prefix: str = "[recording]-"
):
    results = []

    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        if not path.name.startswith(prefix):
            continue

        stat = path.stat()

        created = datetime.fromtimestamp(stat.st_ctime)

        results.append({
            "name": path.name,
            "relative_path": path.relative_to(root),
            "created": created,
        })

    return sorted(results, key=lambda x: x["created"], reverse=True)