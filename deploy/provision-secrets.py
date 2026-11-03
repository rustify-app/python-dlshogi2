#!/usr/bin/env python3
"""Create the Secrets for deploy/k8s in Karmada, from Infisical. Values are never printed.

  python3 deploy/provision-secrets.py [--dry-run]

  floodgate-csa  CSA_HOST / CSA_PORT / CSA_USER / CSA_PASSWORD for pydlshogi2.csa_client, from
                 Infisical Rustify prod /floodgate (FLOODGATE_*), read with INFISICAL_RUSTIFY_*.
  floodgate-nnue-csa  the same for rustify_nnue (rustify-app/rustify-nnue): FLOODGATE_HOST/PORT
                 plus FLOODGATE_NNUE_USER / FLOODGATE_NNUE_PASSWORD.
  zot-creds      image pull secret for zot, from the self-hosted Infisical "Development
                 Environment Variables" dev /general (ZOT_*), read with INFISICAL_LOVEKAPIBARASAN_*.

The identities come from the ai-agent .env (Google Drive: secret/ai-agent/.env); point
AI_AGENT_ENV at a local copy (default ~/.config/homelab/ai-agent.env). After rotating a value,
re-run and restart: kubectl --context lovekapibarasan-office -n dlshogi rollout restart deploy/floodgate
"""
import argparse, base64, json, os, subprocess, sys, urllib.parse, urllib.request
from pathlib import Path

RUSTIFY_PROJECT = "9336b22e-75ef-4fff-b072-6d414efd2979"
DEV_ENV_PROJECT = "d2416e7a-c03e-46b0-bb5a-9a13cc597046"
NAMESPACE, LABELS = "dlshogi", {"app.kubernetes.io/part-of": "dlshogi"}
CONTEXT = os.getenv("KARMADA_CONTEXT", "karmada-apiserver")


def load_env(path):
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def fetch(prefix, project, env, path):
    base = os.environ[f"INFISICAL_{prefix}_ENDPOINT"].rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "https://" + ("app.infisical.com" if base == "infisical.com" else base)
    hdr = {"User-Agent": "python-httpx/0.27", "Content-Type": "application/json"}

    def call(url, body=None, extra=None):
        req = urllib.request.Request(base + url, headers={**hdr, **(extra or {})},
                                     data=json.dumps(body).encode() if body else None)
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)

    token = call("/api/v1/auth/universal-auth/login", {
        "clientId": os.environ[f"INFISICAL_{prefix}_CLIENT_ID"],
        "clientSecret": os.environ[f"INFISICAL_{prefix}_CLIENT_SECRET"]})["accessToken"]
    q = urllib.parse.urlencode({"workspaceId": project, "environment": env, "secretPath": path})
    return {s["secretKey"]: s["secretValue"]
            for s in call(f"/api/v3/secrets/raw?{q}", extra={"Authorization": f"Bearer {token}"})["secrets"]}


def secret(name, data, kind="Opaque"):
    return {"apiVersion": "v1", "kind": "Secret", "type": kind,
            "metadata": {"name": name, "namespace": NAMESPACE, "labels": LABELS},
            "data": {k: base64.b64encode(v.encode()).decode() for k, v in data.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    env_file = Path(os.getenv("AI_AGENT_ENV", "~/.config/homelab/ai-agent.env")).expanduser()
    if not env_file.is_file():
        sys.exit(f"{env_file} not found. Download secret/ai-agent/.env from Google Drive and set AI_AGENT_ENV.")
    load_env(env_file)

    fg = fetch("RUSTIFY", RUSTIFY_PROJECT, "prod", "/floodgate")
    csa = {f"CSA_{k}": fg[f"FLOODGATE_{k}"] for k in ("HOST", "PORT", "USER", "PASSWORD")}
    nnue = {**csa, "CSA_USER": fg["FLOODGATE_NNUE_USER"], "CSA_PASSWORD": fg["FLOODGATE_NNUE_PASSWORD"]}
    dev = fetch("LOVEKAPIBARASAN", DEV_ENV_PROJECT, "dev", "/general")
    registry = urllib.parse.urlparse(dev["ZOT_ENDPOINT"]).netloc or dev["ZOT_ENDPOINT"]
    auth = base64.b64encode(f"{dev['ZOT_USERNAME']}:{dev['ZOT_PASSWORD']}".encode()).decode()
    print(f"floodgate-csa: {', '.join(sorted(csa))}\nfloodgate-nnue-csa: {', '.join(sorted(nnue))}\nzot-creds: registry {registry}")
    if args.dry_run:
        return 0
    items = {"apiVersion": "v1", "kind": "List", "items": [
        secret("floodgate-csa", csa),
        secret("floodgate-nnue-csa", nnue),
        secret("zot-creds", {".dockerconfigjson": json.dumps({"auths": {registry: {"auth": auth}}})},
               "kubernetes.io/dockerconfigjson")]}
    return subprocess.run(["kubectl", "--context", CONTEXT, "apply", "-f", "-"],
                          input=json.dumps(items), text=True).returncode


if __name__ == "__main__":
    sys.exit(main())
