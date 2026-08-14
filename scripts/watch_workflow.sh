#!/usr/bin/env bash
# Watch a background Claude Code workflow from a terminal, since the VS Code
# chat pane has no /workflows viewer. Pass a run id, or omit for the newest run.
set -euo pipefail
BASE="$HOME/.claude/projects/-Users-mattjoslin-Documents-Git-Projects-Mammotion-HA"
RUN="${1:-}"
DIR=$( [ -n "$RUN" ] \
  && find "$BASE" -type d -name "$RUN" -print -quit \
  || find "$BASE" -type d -name 'wf_*' -exec stat -f '%m %N' {} + | sort -rn | head -1 | cut -d' ' -f2- )
[ -n "$DIR" ] || { echo "no workflow run found under $BASE"; exit 1; }
echo "watching $(basename "$DIR")"
while :; do
  printf '\033[H\033[2J'
  echo "workflow $(basename "$DIR")   $(date '+%H:%M:%S')"
  echo "---------------------------------------------------------------"
  python3 - "$DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
started = done = 0
for line in open(os.path.join(d, 'journal.jsonl'), errors='replace'):
    try: e = json.loads(line)
    except Exception: continue
    t = str(e.get('type', ''))
    if 'start' in t: started += 1
    if 'result' in t: done += 1
print(f"agents started {started}   completed {done}   running {started-done}")
print()
for f in sorted(os.listdir(d)):
    if not f.endswith('.jsonl') or f == 'journal.jsonl': continue
    p = os.path.join(d, f)
    size = os.path.getsize(p)
    age = int(os.path.getmtime(p))
    print(f"  {f[6:23]:20s} {size/1024:8.0f} KB")
PY
  echo "---------------------------------------------------------------"
  echo "Ctrl-C to stop"
  sleep 5
done
