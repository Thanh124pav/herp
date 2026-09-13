#!/usr/bin/env bash
# Clone (or resync) the third-party baselines to the exact commits recorded
# in docs/v3/external_baselines.lock.json. Skip anything already checked out
# at the correct SHA. Idempotent; safe to re-run.
#
# Also fetches ManiSkill's official baselines (PPO/SAC/TD-MPC2) via sparse
# checkout — required by scripts/tdmpc2_official.py and the SAC parity tests.
#
# Usage:
#     ./scripts/fetch_third_party.sh              # fetch all
#     ONLY="BRO MaxInfoRL" ./scripts/fetch_third_party.sh
#     SKIP="RFCL ActiveRL" ./scripts/fetch_third_party.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOCK="docs/v3/external_baselines.lock.json"
if [[ ! -f "$LOCK" ]]; then
    echo "ERROR: missing lock file $LOCK" >&2; exit 2
fi

MANISKILL_SHA="62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3"
mkdir -p third_party

_want() {
    local name="$1"
    [[ -n "${ONLY:-}" ]] && ! echo " $ONLY " | grep -q " $name " && return 1
    [[ -n "${SKIP:-}" ]] && echo " $SKIP " | grep -q " $name " && return 1
    return 0
}

_fetch() {
    local name="$1" url="$2" path="$3" sha="$4"
    if ! _want "$name"; then echo "[fetch] $name — skipped"; return; fi
    if [[ -d "$path/.git" ]]; then
        local have="$(git -C "$path" rev-parse HEAD 2>/dev/null || echo none)"
        if [[ "$have" == "$sha" ]]; then
            echo "[fetch] $name — already at $sha"; return
        fi
        echo "[fetch] $name — resyncing to $sha"
        git -C "$path" fetch --depth 50 origin
        git -C "$path" checkout -q "$sha"
    else
        echo "[fetch] $name — cloning $url"
        git clone --filter=blob:none "$url" "$path"
        git -C "$path" checkout -q "$sha"
    fi
}

# Parse lock JSON and iterate.
while IFS=$'\t' read -r name url path sha; do
    [[ -z "$name" ]] && continue
    _fetch "$name" "$url" "$path" "$sha"
done < <(python3 -c "
import json, sys
for e in json.load(open('$LOCK')):
    print(f\"{e['name']}\t{e['url']}\t{e['path']}\t{e['commit']}\")
")

# ManiSkill upstream baselines (PPO/SAC/TD-MPC2) — sparse checkout so we
# only pull the examples/baselines tree, not the whole benchmark. Required by
# scripts/tdmpc2_official.py::prepare() and tests/test_native_baselines.py.
if _want ManiSkill; then
    if [[ -d third_party/ManiSkill/.git ]]; then
        have="$(git -C third_party/ManiSkill rev-parse HEAD 2>/dev/null || echo none)"
        if [[ "$have" != "$MANISKILL_SHA" ]]; then
            echo "[fetch] ManiSkill — resyncing to $MANISKILL_SHA"
            git -C third_party/ManiSkill fetch --depth 50 origin
            git -C third_party/ManiSkill checkout -q "$MANISKILL_SHA"
        else
            echo "[fetch] ManiSkill — already at $MANISKILL_SHA"
        fi
    else
        echo "[fetch] ManiSkill — cloning (sparse)"
        git clone --filter=blob:none --sparse \
            https://github.com/mani-skill/ManiSkill.git third_party/ManiSkill
        git -C third_party/ManiSkill sparse-checkout set \
            examples/baselines/ppo \
            examples/baselines/sac \
            examples/baselines/tdmpc2
        git -C third_party/ManiSkill checkout -q "$MANISKILL_SHA"
    fi
fi

echo "[fetch] done — third_party/ populated"
ls -la third_party/
