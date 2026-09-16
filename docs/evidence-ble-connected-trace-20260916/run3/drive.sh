#!/bin/zsh
cd "/Users/mattjoslin/Documents/Git Projects/Mammotion-HA"
set -a; . ./.env; set +a
OUT=${OUT:-docs/evidence-ble-connected-trace-20260916}
mkdir -p $OUT
PY=.venv/bin/python
st() { curl -s -H "Authorization: Bearer $HA_TOKEN" "$HA_URL/api/states/$1" | python3 -c "import json,sys;print(json.load(sys.stdin)['state'])"; }
BASE_ERR=$(curl -s -H "Authorization: Bearer $HA_TOKEN" "$HA_URL/api/states/sensor.back_yard_clip_skywalker_last_error_time" | python3 -c "import json,sys;print(json.load(sys.stdin)['state'])")
legs=("S1 4.94 -4.82" "S2 4.94 -5.82" "S3 4.94 -6.82" "S4 4.94 -5.82" "S5 4.94 -4.82" "S6 4.94 -3.82" "S7 4.94 -2.82" "S8 4.94 -1.82" "S9 4.94 -2.82" "S10 4.94 -3.82" "S11 4.94 -4.82" "S12 4.94 -5.82")
START=${1:-1}
idx=0
for spec in "${legs[@]}"; do
  idx=$((idx+1)); (( idx < START )) && continue
  parts=(${=spec}); L=$parts[1]; TX=$parts[2]; TY=$parts[3]
  tries=0
  while true; do
  tries=$((tries+1))
  if [[ $L == S5 || $L == S10 ]]; then echo "$(date -u +%T) wait 60s after turn leg"; sleep 60; fi
  B=$(st sensor.back_yard_clip_skywalker_battery); R=$(st sensor.back_yard_clip_skywalker_rtk_position)
  E=$(curl -s -H "Authorization: Bearer $HA_TOKEN" "$HA_URL/api/states/sensor.back_yard_clip_skywalker_last_error_time" | python3 -c "import json,sys;print(json.load(sys.stdin)['state'])")
  echo "$(date -u +%T) $L pre: battery=$B rtk=$R err_time=$E"
  if (( B < 30 )) || [[ $R != fix ]] || [[ $E != $BASE_ERR ]]; then echo "STOP_RULE $L battery/rtk/fault"; exit 3; fi
  $PY scripts/ha_set_experimental_motion.py on --yes > $OUT/arm_$L.txt 2>&1
  echo "$(date -u +%T) $L dispatch start" | tee -a $OUT/dispatch_windows.log
  $PY scripts/phase1_leg_runner.py $L $TX $TY scored --out $OUT ${LOWSUN:+--allow-low-sun} ${VIODIP:+--allow-recovered-vio-dip} > $OUT/log_$L.txt 2>&1
  rc=$?
  echo "$(date -u +%T) $L dispatch end rc=$rc" | tee -a $OUT/dispatch_windows.log
  $PY scripts/ha_set_experimental_motion.py off > $OUT/disarm_$L.txt 2>&1
  $PY scripts/ha_set_experimental_motion.py status > $OUT/disarm_verify_$L.txt 2>&1
  if ! grep -q "enabled             : False" $OUT/disarm_verify_$L.txt; then echo "GATE NOT DISARMED after $L"; $PY scripts/ha_set_experimental_motion.py off; exit 4; fi
  tail -3 $OUT/log_$L.txt
  if (( rc == 0 )); then break; fi
  if (( tries < 4 )) && grep -qE "${RETRY_RE:-vio_tracked_features min|ble_client_not_connected}" $OUT/log_$L.txt; then
    cp $OUT/log_$L.txt $OUT/log_${L}_halt$tries.txt
    echo "$(date -u +%T) $L predeclared auto-retry ($tries) after 60s"; sleep 60; continue
  fi
  echo "HALT at $L rc=$rc"; exit 2
  done
done
echo "ALL 12 DONE"
