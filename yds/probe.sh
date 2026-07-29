#! /usr/bin/env bash
# Measure what a DataSphere container can actually do, so we know whether the agentic harness can
# run there and on what terms. One job, everything at once: no second round trip.
#
# The questions, in order of how much they matter:
#   1. can we execute the agent's tool calls in any kind of sandbox (docker, or namespaces)?
#   2. how fast is the CPU? workload 01's whole calibration is a timing threshold, and it was
#      measured at 6.6x-too-slow on the V100 box. A different CPU moves that.
#   3. does the GPU support bf16? if so, running there forks the dtype from every episode on disk.
#   4. is there network egress, and is anything already cached?
set -u

OUT=probe.txt
: > "$OUT"          # declared outputs must exist, always -- a missing one aborts the whole upload
: > probe.json

say() { echo "$@" | tee -a "$OUT"; }
run() { echo "\$ $*" >> "$OUT"; eval "$@" >> "$OUT" 2>&1; echo >> "$OUT"; }

say "=== identity and privileges ==="
run "id"
run "whoami"
run "cat /proc/self/status | grep -E 'CapEff|CapPrm|NoNewPrivs|Seccomp'"
run "capsh --print 2>/dev/null | head -20 || echo 'capsh absent'"

say ""
say "=== docker: is it there, and can we reach a daemon? ==="
run "which docker podman nerdctl 2>&1"
run "ls -la /var/run/docker.sock 2>&1"
run "test -S /var/run/docker.sock && echo 'docker socket present' || echo 'no docker socket'"
run "timeout 20 docker info 2>&1 | head -20 || echo 'docker info failed'"

say ""
say "=== docker-in-docker: could we install and start one? ==="
run "test -e /dev/fuse && echo '/dev/fuse present' || echo 'no /dev/fuse'"
run "cat /proc/filesystems | grep -E 'overlay|fuse' || echo 'no overlay/fuse in /proc/filesystems'"
run "cat /sys/fs/cgroup/cgroup.controllers 2>/dev/null || echo 'no cgroup v2 controllers file'"
run "stat -fc %T /sys/fs/cgroup 2>&1"

say ""
say "=== namespace isolation, the fallback if docker is out ==="
run "which unshare bwrap firejail 2>&1"
run "cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null || echo 'unprivileged_userns_clone unset'"
run "cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo 'max_user_namespaces unset'"
run "unshare --user --map-root-user echo 'userns OK' 2>&1"
run "unshare --user --map-root-user --net echo 'userns+net OK' 2>&1"
run "unshare --user --map-root-user --mount --pid --fork echo 'userns+mount+pid OK' 2>&1"

say ""
say "=== cpu: the number workload 01 depends on ==="
run "nproc"
run "grep -m1 'model name' /proc/cpuinfo"
run "grep -c processor /proc/cpuinfo"
run "cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo 'no cpu.max'"
python3 - >> "$OUT" 2>&1 <<'PY'
import timeit, json, os
data = list(range(100000))
best = min(timeit.repeat(lambda: sum(data), number=10, repeat=20)) / 10
print("sum(list) of 100000 ints: %.6f s" % best)
print("workload 01 target:       0.000100 s")
print("ratio: %.1fx too slow" % (best / 0.0001))
print("V100 box measured:        0.000663 s (6.6x)")
print("affinity cpus: %d" % len(os.sched_getaffinity(0)))
json.dump({"sum_seconds": best, "ratio": best / 0.0001,
           "cpus": os.cpu_count(), "affinity": len(os.sched_getaffinity(0))},
          open("probe.json", "w"), indent=2)
PY

say ""
say "=== gpu ==="
run "nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv 2>&1"
python3 - >> "$OUT" 2>&1 <<'PY'
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported(including_emulation=False))
    else:
        print("no cuda visible")
except Exception as exc:
    print("torch unavailable:", exc)
PY

say ""
say "=== memory, disk, shm ==="
run "free -g | head -2"
run "df -h / /tmp /dev/shm 2>&1 | head -6"
run "cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 'no memory.max'"

say ""
say "=== python, pytest, and what is installed ==="
run "python3 --version"
run "python3 -m pytest --version 2>&1 | head -2"
run "python3 -c 'import sys; print(sys.executable)'"

say ""
say "=== network egress and cache ==="
run "timeout 15 python3 -c \"import urllib.request; print(urllib.request.urlopen('https://huggingface.co', timeout=10).status)\" 2>&1"
run "du -sh \$HF_HOME 2>/dev/null || echo 'HF_HOME unset or empty'"
run "ls ~/.cache/huggingface/hub 2>/dev/null | head || echo 'no hf cache'"

say ""
say "=== can we actually run a subprocess sandbox? ==="
mkdir -p /tmp/sbx && cd /tmp/sbx
cat > t_demo.py <<'PY'
def test_pass():
    assert 1 == 1
def test_fail():
    assert 2 == 3
PY
run "timeout 60 python3 -m pytest -q t_demo.py 2>&1 | tail -3"
cd - > /dev/null

say ""
say "=== done ==="
wc -l "$OUT"
