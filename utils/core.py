from pathlib import Path
from datetime import datetime

# Helper function: Simply find directories and subdirectories with a provided prefix
def find_recording_dirs(
    root: Path,
    prefix: str = "_recording_",
    recursive: bool = True
):
    results = []

    # Choose iterator based on recursion flag
    iterator = root.rglob("*") if recursive else root.iterdir()

    for path in iterator:
        if not path.is_dir():
            continue

        if prefix is not None and not path.name.startswith(prefix):
            continue

        stat = path.stat()
        created = datetime.fromtimestamp(stat.st_ctime)

        results.append({
            "name": path.name,
            "absolute_path": path,
            "relative_path": path.relative_to(root),
            "created": created,
        })

    return sorted(results, key=lambda x: x["created"], reverse=True)

# Helper function: Determine if a provided path contains the listed filenames
def check_valid_dir_with_file(d, filenames):
    base = Path(d)
    return all((base / f).is_file() for f in filenames)

# Helper function: Given a set of directories, confirm if they are valid by checking for filenames immediately within them.
def check_multiple_dirs_with_files(dirs, filenames):
    filenames = set(filenames)
    ok, missing = [], []
    for rec in dirs:
        if check_valid_dir_with_file(rec['absolute_path'], filenames):
            ok.append(rec)
        else:
            missing.append(rec)
    return ok, missing