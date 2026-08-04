#!/usr/bin/env bash

acceptance_exec_immutable() {
  local project_root=$1 output_root=$2 source_script=$3
  shift 3

  if [[ "${ZCP_LAUNCHER_SNAPSHOT_ACTIVE:-}" == 1 ]]; then
    return 0
  fi

  local commit dirty snapshot_dir timestamp bundle source_root project_path source_path relative snapshot sha256
  commit=$(git -C "$project_root" rev-parse HEAD)
  dirty=$(git -C "$project_root" status --porcelain)
  if [[ -n "$dirty" ]]; then
    echo "Project worktree must be clean before an acceptance launcher is snapshotted" >&2
    return 2
  fi

  project_path=$(realpath "$project_root")
  source_path=$(realpath "$source_script")
  case "$source_path" in
    "$project_path"/*) relative=${source_path#"$project_path"/} ;;
    *)
      echo "Acceptance launcher must be tracked inside the project root: $source_path" >&2
      return 2
      ;;
  esac

  timestamp=$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%z)
  snapshot_dir=$output_root/launcher-snapshots
  bundle=$snapshot_dir/$(basename "${source_script%.sh}")-${commit:0:12}-$timestamp-$$
  source_root=$bundle/source
  mkdir -p "$source_root"
  git -C "$project_root" archive "$commit" | tar -x -C "$source_root"
  snapshot=$source_root/$relative
  [[ -f "$snapshot" ]] || {
    echo "Launcher is not present in commit $commit: $relative" >&2
    return 2
  }
  sha256=$(sha256sum "$snapshot" | awk '{print $1}')
  printf '%s  %s\n' "$sha256" "$(basename "$snapshot")" > "$snapshot.sha256"
  chmod -R a-w "$source_root"

  exec env \
    ZCP_LAUNCHER_SNAPSHOT_ACTIVE=1 \
    ZCP_LAUNCHER_SOURCE="$source_script" \
    ZCP_LAUNCHER_SNAPSHOT="$snapshot" \
    ZCP_LAUNCHER_COMMIT="$commit" \
    ZCP_ACCEPTANCE_ROOT="$output_root" \
    ZCP_PROJECT_ROOT="$source_root" \
    bash "$snapshot" "$@"
}
