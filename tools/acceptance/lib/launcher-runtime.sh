#!/usr/bin/env bash

acceptance_exec_immutable() {
  local project_root=$1 output_root=$2 source_script=$3
  shift 3

  if [[ "${ZCP_LAUNCHER_SNAPSHOT_ACTIVE:-}" == 1 ]]; then
    return 0
  fi

  local commit dirty snapshot_dir timestamp snapshot sha256
  commit=$(git -C "$project_root" rev-parse HEAD)
  dirty=$(git -C "$project_root" status --porcelain)
  if [[ -n "$dirty" ]]; then
    echo "Project worktree must be clean before an acceptance launcher is snapshotted" >&2
    return 2
  fi

  timestamp=$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%z)
  snapshot_dir=$output_root/launcher-snapshots
  snapshot=$snapshot_dir/$(basename "${source_script%.sh}")-${commit:0:12}-$timestamp-$$.sh
  mkdir -p "$snapshot_dir"
  install -m 0444 "$source_script" "$snapshot"
  sha256=$(sha256sum "$snapshot" | awk '{print $1}')
  printf '%s  %s\n' "$sha256" "$(basename "$snapshot")" > "$snapshot.sha256"
  chmod 0444 "$snapshot.sha256"

  exec env \
    ZCP_LAUNCHER_SNAPSHOT_ACTIVE=1 \
    ZCP_LAUNCHER_SOURCE="$source_script" \
    ZCP_LAUNCHER_SNAPSHOT="$snapshot" \
    ZCP_LAUNCHER_COMMIT="$commit" \
    ZCP_PROJECT_ROOT="$project_root" \
    bash "$snapshot" "$@"
}
