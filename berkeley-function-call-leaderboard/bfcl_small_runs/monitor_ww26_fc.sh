#!/usr/bin/env bash
set -euo pipefail

BFCL_ROOT=/home/sstrehlk/src/llm-acc-check-gorilla/gorilla/berkeley-function-call-leaderboard
GPT_RUN=bfcl_fc_ww26_gpt_limit50_cpu_20260703_105833
QWEN_RUN=bfcl_fc_ww26_qwen_limit50_cpu_20260703_105833

print_run() {
    local label=$1
    local pattern=$2
    local result_dir=$3
    local score_dir=$4
    local log_file=$5
    local pid pcpu pmem rss_kb elapsed state files scores rss_gib cpu_threads progress last_id

    pid=$(ps -ww -eo pid,args | awk -v pattern="$pattern" '$0 ~ /python/ && $0 ~ /\/bin\/bfcl/ && index($0, pattern) { print $1; exit }')
    if [[ -d "$result_dir" ]]; then
        files=$(find "$result_dir" -type f | wc -l)
    else
        files=0
    fi
    if [[ -d "$score_dir" ]]; then
        scores=$(find "$score_dir" -type f | wc -l)
    else
        scores=0
    fi
    progress=$(grep -aoE '[0-9]+/50' "$log_file" 2>/dev/null | tail -n 1 || true)
    last_id=$(grep -aoE 'ID: base_[0-9]+, Turn: [0-9]+, Step: [0-9]+' "$log_file" 2>/dev/null | tail -n 1 | sed -E 's/^ID: base_([0-9]+), Turn: ([0-9]+), Step: ([0-9]+)/b\1\/t\2\/s\3/' || true)
    [[ -z "$progress" ]] && progress="-"
    [[ -z "$last_id" ]] && last_id="-"

    if [[ -z "$pid" ]]; then
        printf '%-5s %-7s %-8s %-8s %-8s %-10s %-8s %-8s %-8s %-10s\n' "$label" "-" "done?" "-" "-" "-" "$files" "$scores" "$progress" "$last_id"
        return
    fi

    read -r state pcpu pmem rss_kb elapsed < <(ps -p "$pid" -o stat=,pcpu=,pmem=,rss=,etime=)
    rss_gib=$(awk -v kb="$rss_kb" 'BEGIN { printf "%.1fGiB", kb / 1024 / 1024 }')
    cpu_threads=$(awk -v cpu="$pcpu" 'BEGIN { printf "%.1f", cpu / 100 }')
    printf '%-5s %-7s %-8s %-8s %-8s %-10s %-8s %-8s %-8s %-10s\n' "$label" "$pid" "$state" "$cpu_threads" "$pmem%" "$rss_gib" "$files" "$scores" "$progress" "$last_id"
}

cd "$BFCL_ROOT"
date -Is
echo
printf '%-5s %-7s %-8s %-8s %-8s %-10s %-8s %-8s %-8s %-10s\n' "run" "pid" "state" "cpuThr" "mem%" "rss" "resultF" "scoreF" "prog" "lastID"
printf '%-5s %-7s %-8s %-8s %-8s %-10s %-8s %-8s %-8s %-10s\n' "-----" "-------" "--------" "--------" "--------" "----------" "--------" "--------" "--------" "----------"
print_run "gpt" "$GPT_RUN" "$BFCL_ROOT/bfcl_small_runs/results/$GPT_RUN" "$BFCL_ROOT/bfcl_small_runs/scores/$GPT_RUN" "$BFCL_ROOT/bfcl_small_runs/logs/$GPT_RUN.log"
print_run "qwen" "$QWEN_RUN" "$BFCL_ROOT/bfcl_small_runs/results/$QWEN_RUN" "$BFCL_ROOT/bfcl_small_runs/scores/$QWEN_RUN" "$BFCL_ROOT/bfcl_small_runs/logs/$QWEN_RUN.log"

echo
free -h | awk 'NR == 1 || NR == 2 || NR == 3 { print }'
echo
uptime
echo
tmux ls 2>/dev/null | grep -E "${GPT_RUN}|${QWEN_RUN}|bfcl_fc_ww26_monitor_30s" || true