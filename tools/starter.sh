#!/usr/bin/env bash
# starter.sh - launches one of the dev/data tools.
#
# Usage:
#   ./tools/starter.sh streamlit       # dataset viewer (browser UI)
#   ./tools/starter.sh label-studio    # annotation server
#
# Always runs from the project root (bird_count/), regardless of CWD.

set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
    echo "Usage: $0 {streamlit|label-studio}"
}

cmd="${1:-}"
case "$cmd" in
    streamlit)
        exec streamlit run tools/visualize_data.py
        ;;
    label-studio | label_studio)
        exec label-studio start --data-dir ../data
        ;;
    -h | --help | "")
        usage
        [[ -z "$cmd" ]] && exit 1 || exit 0
        ;;
    *)
        echo "Unknown command: $cmd" >&2
        usage >&2
        exit 2
        ;;
esac
