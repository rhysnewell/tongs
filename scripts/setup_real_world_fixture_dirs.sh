#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$repo_root/external/protein_medium/queries"

cat <<'EOF'
Created external/protein_medium/queries

Expected files:
  external/protein_medium/uniprot_UP000005640_human.fasta.gz
  external/protein_medium/uniprot_UP000005640_human.fasta
  external/protein_medium/queries/sp_O43739_CYH3_HUMAN.fa

The proteome is the 20,659-sequence download from https://www.uniprot.org/proteomes/UP000005640.
C jackhmmer needs the uncompressed copy to rewind between rounds.
EOF
