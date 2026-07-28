"""Run one steering shard and save its results as the declared job output."""

import argparse
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
output = Path(parser.parse_args().output)
args = os.environ.get("STEERING_ARGS")
if not args:
    raise SystemExit("Set STEERING_ARGS, for example: --benchmark math_500 --limit 1")

result = subprocess.run([sys.executable, "-u", "02_steering/steer.py", *shlex.split(args)])
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in Path("02_steering/results").rglob("*"):
        if path.is_file():
            archive.write(path, path)
raise SystemExit(result.returncode)
