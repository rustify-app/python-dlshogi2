#!/bin/bash
# Waits for /workspace/out/DONE on vast-r4, downloads out/ (verified), then destroys the instance.
# If the instance stops on its own (self-stop / deadline), leaves it alone and exits.
ID=${VAST_INSTANCE_ID:?}; DEST=$HOME/Documents/python-dlshogi2/model/r4; V="python3 -I $HOME/.local/share/homelab-agent/vast.py"
mkdir -p "$DEST"
while true; do
  st=$($V GET /instances/$ID/ 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["instances"].get("actual_status"))' 2>/dev/null)
  if [ -n "$st" ] && [ "$st" != "running" ] && [ "$st" != "loading" ] && [ "$st" != "None" ]; then echo "instance is $st; not touching it"; exit 0; fi
  ssh -o ConnectTimeout=15 vast-r4 'test -f /workspace/out/DONE' 2>/dev/null && break
  sleep 300
done
for t in $(seq 1 20); do rsync -a --partial --timeout=120 -e ssh vast-r4:/workspace/out/ "$DEST/" && break; sleep 20; done
remote=$(ssh vast-r4 'cd /workspace/out && find . -type f ! -name FETCHED -exec sha256sum {} + | sort -k2')
local_=$(cd "$DEST" && find . -type f ! -name FETCHED -exec sha256sum {} + | sort -k2)
if [ "$remote" = "$local_" ]; then
  ssh vast-r4 'touch /workspace/out/FETCHED'
  $V DELETE /instances/$ID/ && echo "fetched to $DEST and destroyed instance $ID"
else
  echo "checksum mismatch; instance left running (it stops itself within 3h)"
fi
