#!/bin/zsh
# Restart Home Assistant Core and wait for the API to come back.
#
# Usage:  scripts/ha_restart.sh
# Requires HA_TOKEN in the environment:  set -a && source .env && set +a
#
# A restart is REQUIRED for changed integration Python to take effect — a
# config-entry reload does not reload modules. Takes roughly 60-135s.
set -u
: "${HA_TOKEN:?HA_TOKEN not set (run: set -a && source .env && set +a)}"
HOST="http://192.168.1.106:8123"

echo "==> requesting restart"
code=$(curl -s -o /dev/null -w "%{http_code}" -m 30 -X POST \
  -H "Authorization: Bearer $HA_TOKEN" -H "Content-Type: application/json" \
  "$HOST/api/services/homeassistant/restart")
echo "    HTTP $code"
[[ "$code" == "200" ]] || { echo "restart request failed"; exit 1; }

echo "==> waiting for API"
start=$(date +%s)
for i in $(seq 1 90); do
  sleep 5
  if curl -s -m 5 -H "Authorization: Bearer $HA_TOKEN" "$HOST/api/" 2>/dev/null | grep -q "API running"; then
    echo "    API up after $(( $(date +%s) - start ))s"
    # The integration keeps loading after the API answers; entities trickle in.
    for j in $(seq 1 60); do
      n=$(curl -s -m 10 -H "Authorization: Bearer $HA_TOKEN" "$HOST/api/states" 2>/dev/null \
          | python3 -c "import json,sys; print(sum(1 for s in json.load(sys.stdin) if 'skywalker' in s['entity_id']))" 2>/dev/null || echo 0)
      if [[ "${n:-0}" -ge 100 ]]; then
        echo "    mammotion entities: $n  (total $(( $(date +%s) - start ))s)"
        exit 0
      fi
      sleep 5
    done
    echo "    WARNING: API up but mammotion entities did not reach 100 — check the integration"
    exit 1
  fi
done
echo "    TIMEOUT: API did not come back"
exit 1
