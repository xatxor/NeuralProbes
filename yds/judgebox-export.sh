#! /usr/bin/env bash
# Move the baseline verdicts from the rented judging box to the long-lived server, directly.
#
# Direct, not via this Mac. Both machines are in datacentres and the hop between them is far faster
# than pulling ~170 MB down a domestic link and pushing it back up; routing through here also puts a
# copy on a laptop that does not want it.
#
# Authorisation without moving any existing credential: a throwaway keypair is generated ON THE
# SERVER and its public half is added to the box's authorized_keys. The box is destroyed when the
# rental ends, so the grant dies with it. The rental's own .pem never leaves this Mac, and the
# server's password never reaches the box.
#
# The verification is not decoration. A judge run can exit 0 having written almost nothing: at a
# 512-token reply cap, 245 of 248 smoke verdicts came back empty while the process reported success.
#
# Usage:  HOST=ubuntu@1.2.3.4 KEY=~/Downloads/box.pem SRV_PASS=... ./judgebox-export.sh
set -uo pipefail

: "${HOST:?set HOST, e.g. ubuntu@1.2.3.4}"
: "${KEY:?set KEY, the rental .pem}"
: "${SRV_PASS:?set SRV_PASS for User18@176.109.111.31}"
root=${ROOT:-judge}
remote=${REMOTE_DIR:-verdicts-plus_baseline-minus_baseline}
expect=${EXPECT:-168576}
SRV=${SRV:-User18@176.109.111.31}
dest=${DEST:-neuralprobes/verdict-baseline}

BOXSSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes)
SRVSSH=(-o StrictHostKeyChecking=no -o PubkeyAuthentication=no
        -o PreferredAuthentications=keyboard-interactive -o ConnectTimeout=25)
srv() { sshpass -p"$SRV_PASS" ssh "${SRVSSH[@]}" "$SRV" "$@"; }

echo "=== waiting for judging to finish ==="
# One short connection per check. This class of provider drops long-lived sessions, and a chain left
# hanging on one dies silently with it.
for _ in $(seq 1 480); do
    alive=$("${BOXSSH[@]}" "$HOST" 'pgrep -cf "bin/python judge"' 2>/dev/null) || { sleep 30; continue; }
    [ "${alive:-1}" = "0" ] && break
    sleep 30
done

echo "=== what the judge reported ==="
"${BOXSSH[@]}" "$HOST" "grep -ahE 'wrote .*judgements' ~/$root/judge-*.log 2>/dev/null"

echo "=== granting the server one-way access to the box ==="
pub=$(srv 'test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q; cat ~/.ssh/id_ed25519.pub')
[ -n "$pub" ] || { echo "could not read the server's public key" >&2; exit 1; }
"${BOXSSH[@]}" "$HOST" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
    grep -qF '$pub' ~/.ssh/authorized_keys 2>/dev/null || echo '$pub' >> ~/.ssh/authorized_keys; \
    chmod 600 ~/.ssh/authorized_keys" || { echo "could not authorise the server on the box" >&2; exit 1; }

echo "=== server pulls from box ==="
box_host=${HOST#*@}
srv "mkdir -p ~/$dest && rsync -az --stats \
        -e 'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25' \
        '$HOST:~/$root/$remote/' ~/$dest/ 2>&1 | tail -6" \
    || { echo "server-side rsync FAILED" >&2; exit 1; }

for f in judge-0.log judge-1.log setup.log versions.txt; do
    srv "rsync -az -e 'ssh -o StrictHostKeyChecking=no' '$HOST:~/$root/$f' ~/$dest/ 2>/dev/null" || true
done

echo "=== verifying, on the server ==="
srv "cd ~/$dest && python3 - . $expect" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

out, expect = Path(sys.argv[1]), int(sys.argv[2])
files = sorted(out.rglob("labels.jsonl"))
if not files:
    sys.exit("FAIL: nothing arrived")

total = usable = echoed = 0
kinds: Counter = Counter()
for path in files:
    rows = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            sys.exit(f"FAIL: unparseable line in {path}")
        rows += 1
        total += 1
        usable += bool(row.get("verdict"))
        echoed += bool(row.get("echoed"))
        kinds[row.get("comparison")] += 1
    print(f"  {path.parent.name}: {rows} rows")

print(f"  TOTAL {total}  usable {usable} ({100 * usable / max(1, total):.1f}%)  "
      f"echoed {echoed} ({100 * echoed / max(1, total):.1f}%)")
print(f"  by comparison: {dict(kinds)}")

if total < expect * 0.9:
    sys.exit(f"FAIL: expected ~{expect:,}, got {total:,}")
if usable < total * 0.5:
    sys.exit(f"FAIL: only {100 * usable / max(1, total):.1f}% usable -- check --reply-tokens")
if set(kinds) - {"plus_baseline", "minus_baseline"}:
    print(f"  WARNING: unexpected comparison kinds: {set(kinds)}")
print("VERIFIED")
PY
status=$?

echo "=== on the server now ==="
srv "du -sh ~/$dest; find ~/$dest -name labels.jsonl -exec sha256sum {} \;"

if [ "$status" -ne 0 ]; then
    echo "=== VERIFICATION FAILED; box left running, do not destroy it ===" >&2
    exit 1
fi
echo "=== export complete; verdicts are on the server, box still running ==="
