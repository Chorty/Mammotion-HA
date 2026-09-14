#!/bin/zsh
# Usage: run_leg.sh LABEL TX TY  -- wait for blade latch to clear (<=480 s), arm, run, ALWAYS disarm.
cd "/Users/mattjoslin/Documents/Git Projects/Mammotion-HA" || exit 9
set -a; source .env; set +a
OUT="/private/tmp/claude-501/-Users-mattjoslin-Documents-Git-Projects-Mammotion-HA/1f622a68-5c5a-4ba9-9b7e-0009a0013142/scratchpad/repeat-20260914"
LABEL=$1; TX=$2; TY=$3
.venv/bin/python - <<'EOF'
import os,sys,time,datetime
sys.path.insert(0,'scripts')
from mammotion_ha_helpers import post_service
t0=time.time()
while True:
    b=post_service(os.environ['HA_URL'],os.environ['HA_TOKEN'],'mammotion','export_runtime_state',{'entity_id':'lawn_mower.back_yard_clip_skywalker'},90).get('blade')
    if not b:
        print('empty runtime response; not armed',flush=True); sys.exit(4)
    print(datetime.datetime.now(datetime.UTC).isoformat(),'blade rpm',b['current_cutter_rpm'],'latched',b['blade_rpm_looks_latched'],b['safety_blockers'],flush=True)
    if not b['blade_rpm_looks_latched'] and not b['safety_blockers'] and b['reported_state_label']=='OFF': sys.exit(0)
    if time.time()-t0>60: sys.exit(3)
    time.sleep(20)
EOF
prc=$?
if [[ $prc -ne 0 ]]; then echo "PRECHECK rc=$prc -- not armed"; exit $prc; fi
.venv/bin/python scripts/ha_set_experimental_motion.py on --yes
echo "ARMED $(date -u +%FT%TZ)"
.venv/bin/python scripts/phase1_leg_runner.py "$LABEL" "$TX" "$TY" scored --out "$OUT"
rc=$?
echo "RUNNER rc=$rc $(date -u +%FT%TZ)"
.venv/bin/python scripts/ha_set_experimental_motion.py off --yes
.venv/bin/python scripts/ha_set_experimental_motion.py status
exit $rc
