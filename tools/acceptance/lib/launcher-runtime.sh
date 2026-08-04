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

acceptance_gpu_lock_metadata_path() {
  local lock_path=$1
  printf '%s.lease\n' "$lock_path"
}

_acceptance_lock_iso_now() {
  TZ=Asia/Shanghai date -Is
}

_acceptance_lock_epoch_now() {
  date +%s
}

_acceptance_lock_uuid() {
  local lock_path=$1 identity
  identity=$(basename "$lock_path")
  identity=${identity%.lock}
  if [[ "$identity" == GPU-* ]]; then
    printf '%s\n' "$identity"
  else
    printf '%s\n' "${ZCP_GPU_LOCK_UUID:-}"
  fi
}

_acceptance_lock_write_owner() {
  local lock_path=$1 lock_fd=$2 owner_pid=$3 host=$4 uuid=$5 acquired_at=$6
  : > "$lock_path"
  {
    printf 'pid=%s\n' "$owner_pid"
    printf 'host=%s\n' "$host"
    printf 'uuid=%s\n' "$uuid"
    printf 'acquired_at=%s\n' "$acquired_at"
  } >&"$lock_fd"
}

_acceptance_lock_clear_owner() {
  local lock_path=$1
  : > "$lock_path"
}

_acceptance_lock_write_metadata() {
  local metadata_path=$1 lease_id=$2 owner_pid=$3 owner_start_ticks=$4
  local supervisor_pid=$5 owner_label=$6 acquired_at=$7 acquired_epoch=$8
  local heartbeat_pid=$9 command_text=${10} lease_seconds=${11} host=${12} uuid=${13}
  local heartbeat_at heartbeat_epoch lease_expires temporary
  heartbeat_at=$(_acceptance_lock_iso_now)
  heartbeat_epoch=$(_acceptance_lock_epoch_now)
  lease_expires=$((heartbeat_epoch + lease_seconds))
  temporary=$metadata_path.tmp-$lease_id
  umask 077
  {
    printf 'schema_version=1\n'
    printf 'authority=kernel_flock\n'
    printf 'lease_enforcement=observability_only_never_unlink_lock\n'
    printf 'state=held\n'
    printf 'lease_id=%s\n' "$lease_id"
    printf 'pid=%s\n' "$owner_pid"
    printf 'host=%s\n' "$host"
    printf 'uuid=%s\n' "$uuid"
    printf 'owner_pid=%s\n' "$owner_pid"
    printf 'owner_start_ticks=%s\n' "$owner_start_ticks"
    printf 'supervisor_pid=%s\n' "$supervisor_pid"
    printf 'heartbeat_pid=%s\n' "$heartbeat_pid"
    printf 'owner_label=%q\n' "$owner_label"
    printf 'command=%q\n' "$command_text"
    printf 'acquired_at=%s\n' "$acquired_at"
    printf 'acquired_epoch=%s\n' "$acquired_epoch"
    printf 'heartbeat_at=%s\n' "$heartbeat_at"
    printf 'heartbeat_epoch=%s\n' "$heartbeat_epoch"
    printf 'lease_seconds=%s\n' "$lease_seconds"
    printf 'lease_expires_epoch=%s\n' "$lease_expires"
  } > "$temporary"
  mv -f "$temporary" "$metadata_path"
}

_acceptance_lock_cleanup_metadata() {
  local metadata_path=$1 lease_id=$2
  if [[ -f "$metadata_path" ]] && grep -Fqx "lease_id=$lease_id" "$metadata_path"; then
    rm -f "$metadata_path"
  fi
  rm -f "$metadata_path.tmp-$lease_id"
}

_acceptance_lock_emit() {
  local event=$1
  shift
  if declare -F supervisor_event >/dev/null 2>&1; then
    supervisor_event "$event" "$@"
  fi
}

acceptance_with_gpu_lock() (
  set +e
  local lock_path=$1 timeout=$2 owner_label=$3
  shift 3
  if [[ $# -eq 0 ]]; then
    echo "acceptance_with_gpu_lock requires a command" >&2
    return 2
  fi

  local heartbeat_interval=${ZCP_GPU_LOCK_HEARTBEAT_SECONDS:-5}
  local lease_seconds=${ZCP_GPU_LOCK_LEASE_SECONDS:-30}
  if [[ ! "$lease_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "ZCP_GPU_LOCK_LEASE_SECONDS must be a positive integer" >&2
    return 2
  fi
  if [[ ! "$heartbeat_interval" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "ZCP_GPU_LOCK_HEARTBEAT_SECONDS must be positive" >&2
    return 2
  fi
  if ! awk -v value="$heartbeat_interval" 'BEGIN { exit !(value > 0) }'; then
    echo "ZCP_GPU_LOCK_HEARTBEAT_SECONDS must be positive" >&2
    return 2
  fi

  local metadata_path lock_fd lease_id owner_pid owner_start_ticks supervisor_pid host uuid
  local acquired_at acquired_epoch heartbeat_pid= task_pid= released=0 command_text
  metadata_path=$(acceptance_gpu_lock_metadata_path "$lock_path")
  mkdir -p "$(dirname "$lock_path")"
  exec {lock_fd}<>"$lock_path"
  if ! flock -w "$timeout" "$lock_fd"; then
    exec {lock_fd}>&-
    _acceptance_lock_emit lock_timeout "gpu_lock=$lock_path owner=$owner_label timeout=$timeout"
    return 4
  fi

  owner_pid=$BASHPID
  owner_start_ticks=$(awk '{print $22}' "/proc/$owner_pid/stat" 2>/dev/null || printf 'unknown')
  supervisor_pid=$$
  host=$(hostname)
  uuid=$(_acceptance_lock_uuid "$lock_path")
  acquired_at=$(_acceptance_lock_iso_now)
  acquired_epoch=$(_acceptance_lock_epoch_now)
  lease_id=$owner_pid-$(date +%s%N)-$RANDOM-$RANDOM
  printf -v command_text '%q ' "$@"

  _acceptance_lock_finish() {
    local exit_code=$1
    if [[ -n "$task_pid" ]] && kill -0 "$task_pid" 2>/dev/null; then
      kill -TERM "$task_pid" 2>/dev/null || true
      wait "$task_pid" 2>/dev/null || true
    fi
    task_pid=
    if [[ -n "$heartbeat_pid" ]]; then
      kill -TERM "$heartbeat_pid" 2>/dev/null || true
      wait "$heartbeat_pid" 2>/dev/null || true
    fi
    heartbeat_pid=
    _acceptance_lock_cleanup_metadata "$metadata_path" "$lease_id"
    if [[ "$released" -eq 0 ]]; then
      _acceptance_lock_clear_owner "$lock_path"
      flock -u "$lock_fd" 2>/dev/null || true
      exec {lock_fd}>&-
      released=1
      _acceptance_lock_emit lock_released \
        "gpu_lock=$lock_path owner=$owner_label holder=$owner_pid exit_code=$exit_code lease_id=$lease_id"
    fi
  }

  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit_code=$?; _acceptance_lock_finish "$exit_code"; exit "$exit_code"' EXIT

  _acceptance_lock_write_owner \
    "$lock_path" "$lock_fd" "$owner_pid" "$host" "$uuid" "$acquired_at"
  (
    exec {lock_fd}>&-
    while sleep "$heartbeat_interval"; do
      _acceptance_lock_write_metadata \
        "$metadata_path" "$lease_id" "$owner_pid" "$owner_start_ticks" \
        "$supervisor_pid" "$owner_label" "$acquired_at" "$acquired_epoch" \
        "$BASHPID" "$command_text" "$lease_seconds" "$host" "$uuid" || exit 1
    done
  ) &
  heartbeat_pid=$!
  _acceptance_lock_write_metadata \
    "$metadata_path" "$lease_id" "$owner_pid" "$owner_start_ticks" \
    "$supervisor_pid" "$owner_label" "$acquired_at" "$acquired_epoch" \
    "$heartbeat_pid" "$command_text" "$lease_seconds" "$host" "$uuid"
  _acceptance_lock_emit lock_acquired \
    "gpu_lock=$lock_path owner=$owner_label holder=$owner_pid lease_id=$lease_id heartbeat=$heartbeat_pid"

  (
    exec {lock_fd}>&-
    "$@"
  ) &
  task_pid=$!
  local exit_code
  if wait "$task_pid"; then
    exit_code=0
  else
    exit_code=$?
  fi
  task_pid=
  _acceptance_lock_finish "$exit_code"
  trap - EXIT INT TERM
  return "$exit_code"
)

acceptance_gpu_lock_observe() {
  local lock_path=$1 metadata_path descriptor held
  metadata_path=$(acceptance_gpu_lock_metadata_path "$lock_path")
  mkdir -p "$(dirname "$lock_path")"
  exec {descriptor}>>"$lock_path"
  if flock -n "$descriptor"; then
    held=false
    flock -u "$descriptor"
  else
    held=true
  fi
  exec {descriptor}>&-
  printf 'flock_held=%s\n' "$held"
  if [[ -f "$metadata_path" ]]; then
    cat "$metadata_path"
  else
    printf 'state=unowned\n'
  fi
}
