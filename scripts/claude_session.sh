#!/usr/bin/env bash
# Carry Claude Code conversations for this project to another machine (optional).
#   ./scripts/claude_session.sh export    copy this project's sessions into data/handoff/claude_session/
#   ./scripts/claude_session.sh import    copy them back into ~/.claude/projects/<this directory> on the new machine
# Then, from the repo root: claude --resume <session id>   (ids are the *.jsonl names).
# CLAUDE_SESSION_DIR overrides the bundle location. Sessions can hold prompts and file contents, so keep
# the bundle out of git (data/ is ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
BUNDLE="${CLAUDE_SESSION_DIR:-$NATSTEER_DATA/handoff/claude_session}"
PROJ="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/projects/$(pwd | sed 's#[^A-Za-z0-9]#-#g')"
case "${1:-}" in
  export) mkdir -p "$BUNDLE"; cp -r "$PROJ"/. "$BUNDLE"/; echo "exported $(ls "$BUNDLE"/*.jsonl | wc -l) sessions to $BUNDLE" ;;
  import) mkdir -p "$PROJ"; cp -rn "$BUNDLE"/. "$PROJ"/ 2>/dev/null || cp -r --update=none "$BUNDLE"/. "$PROJ"/; echo "imported into $PROJ"; ls "$PROJ"/*.jsonl ;;
  *) echo "usage: claude_session.sh export|import"; exit 2 ;;
esac
