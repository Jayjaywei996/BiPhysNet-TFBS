#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "[1/5] Checking hard-coded local paths outside legacy/..."
if grep -RInE 'C:\\Users\\13971\\Music|/home/wjw' "$PROJECT_ROOT" --exclude-dir=.git --exclude-dir=legacy --exclude=check_project.sh --exclude=check_project.ps1; then
  echo "Found old hard-coded paths. Review the matches above."
else
  echo "No old local hard-coded paths found outside legacy/."
fi

echo "[2/5] Checking for private PDFs in GitHub-ready tree..."
if find "$PROJECT_ROOT" -path "$PROJECT_ROOT/.git" -prune -o -iname '*.pdf' -print | grep -q .; then
  echo "PDF files were found. Keep manuscript/patent drafts private before publishing."
  find "$PROJECT_ROOT" -path "$PROJECT_ROOT/.git" -prune -o -iname '*.pdf' -print
else
  echo "No PDF files found."
fi

echo "[3/5] Compiling Python files..."
python -m py_compile "$PROJECT_ROOT"/src/biphysnet/*.py "$PROJECT_ROOT"/tools/*.py

echo "[4/5] Running tests..."
python -m pytest tests

echo "[5/5] Git status..."
if [ ! -d .git ]; then git init >/dev/null; fi
git status --short
