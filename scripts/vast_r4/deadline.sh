#!/bin/bash
# Hard cap: whatever happens, stop this instance 16 hours after start.
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^CONTAINER_(ID|API_KEY)=" | xargs)
vast_stop() { curl -s -m 60 -X PUT -H "Authorization: Bearer $CONTAINER_API_KEY" -H "Content-Type: application/json" -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/"; }
sleep $((16*3600)); cd /workspace; echo "$(date -u +%FT%TZ) deadline reached" >> pipeline.log
for t in 1 2 3; do vast_stop >> pipeline.log 2>&1; sleep 120; done
