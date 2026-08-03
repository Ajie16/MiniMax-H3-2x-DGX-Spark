#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0
report_failure() {
  printf 'public audit failed: %s\n' "$1" >&2
  fail=1
}

for required in README.md LICENSE NOTICE MODEL-LICENSE.md SECURITY.md CONTRIBUTING.md; do
  [[ -f "$required" ]] || report_failure "missing $required"
done

grep -q 'Apache License' LICENSE || report_failure 'LICENSE is not Apache-2.0 text'
grep -q 'MiniMax H3 Community License Agreement' MODEL-LICENSE.md ||
  report_failure 'model-license warning is missing'

for script in scripts/*.sh; do
  [[ -x "$script" ]] || report_failure "script is not executable: $script"
done

if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  report_failure '.env is tracked'
fi

if git ls-files | grep -E '\.(mp4|mov|mkv|avi|jpg|jpeg|png|webp|safetensors|bin|pt|pth|ckpt)$' >/dev/null; then
  git ls-files | grep -E '\.(mp4|mov|mkv|avi|jpg|jpeg|png|webp|safetensors|bin|pt|pth|ckpt)$' >&2
  report_failure 'generated media or model-like binary is tracked'
fi

while IFS= read -r file; do
  [[ -f "$file" ]] || continue
  size="$(stat -c '%s' "$file")"
  if (( size > 5 * 1024 * 1024 )); then
    printf '%s (%s bytes)\n' "$file" "$size" >&2
    report_failure 'tracked file larger than 5 MiB'
  fi
done < <(git ls-files)

credential_pattern='(-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})'
if git grep -nI -E "$credential_pattern" -- ':!scripts/public-audit.sh'; then
  report_failure 'credential-shaped text found'
fi

if git grep -nI 'joeyr1982' -- ':!scripts/public-audit.sh'; then
  report_failure 'stale GitHub owner found'
fi

private_ip_pattern='(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})'
if git grep -nI -E "$private_ip_pattern" -- ':!scripts/public-audit.sh'; then
  report_failure 'private IPv4 address found'
fi

git diff --check || report_failure 'whitespace errors found'
git show --check --format= HEAD || report_failure 'HEAD contains whitespace errors'
git fsck --no-progress >/dev/null || report_failure 'git object check failed'

for script in scripts/*.sh; do
  bash -n "$script" || report_failure "bash syntax failed: $script"
done

python3 - <<'PY' || report_failure 'Python syntax check failed'
from pathlib import Path

for path in sorted(Path('.').glob('**/*.py')):
    if '.git' not in path.parts and '__pycache__' not in path.parts:
        compile(path.read_text(), str(path), 'exec')
PY

(( fail == 0 )) || exit 1
echo 'public audit passed'
