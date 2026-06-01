#!/usr/bin/env bash
# test_fusedops.sh — verify fused-ops wheels from TestPyPI actually work
# Installs fireants-fused-ops-cu<X> from test.pypi.org into isolated conda
# envs, then runs the test_fusedops_*.py test suite against each variant.
# Assumes: CUDA-capable machine with compatible driver, conda on PATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------- full CI matrix (cuda_short:torch:python) ----------
ALL_ENTRIES=(
  "118:2.1.0:3.10"
  "118:2.1.0:3.11"
  "118:2.2.0:3.12"
  "121:2.3.0:3.10"
  "121:2.3.0:3.11"
  "121:2.3.0:3.12"
  "124:2.5.0:3.10"
  "124:2.5.0:3.11"
  "124:2.5.0:3.12"
)
# ---------------------------------------------------------------

PYPI_INDEX="https://test.pypi.org/simple/"
ENTRIES=()
VERSION=""   # optional: pin a specific package version
IMPORT_ONLY=false  # if true, only verify install + import (skip pytest)

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Test fireants-fused-ops wheels from TestPyPI in isolated conda environments.

Options:
  --entry CU:TORCH:PY  Run a single matrix entry, e.g. "124:2.5.0:3.10"
  --all                 Run all 9 matrix entries (default if no --entry given)
  --version VER         Pin fused-ops package version (e.g. "1.2.0")
  --pypi                Install from real PyPI instead of TestPyPI
  --import-only         Only verify install + import (skip running tests)
  -h, --help            Show this help

Environment variable overrides (single-entry shortcut):
  CUDA_SHORT, TORCH, PYTHON   e.g.  CUDA_SHORT=124 TORCH=2.5.0 PYTHON=3.10 $0

Examples:
  $0                                  # full matrix from TestPyPI
  $0 --entry 124:2.5.0:3.10           # single entry
  $0 --version 1.2.0                  # pin version
  $0 --pypi                           # test real PyPI release
  $0 --import-only                    # quick: just verify install + import
  CUDA_SHORT=121 TORCH=2.3.0 PYTHON=3.11 $0
EOF
  exit 0
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --entry)     ENTRIES+=("$2"); shift 2 ;;
    --all)       shift ;;
    --version)   VERSION="$2"; shift 2 ;;
    --pypi)      PYPI_INDEX="https://pypi.org/simple/"; shift ;;
    --import-only) IMPORT_ONLY=true; shift ;;
    -h|--help)   usage ;;
    *)           echo "Unknown option: $1"; usage ;;
  esac
done

# If env vars are set and no --entry was given, build a single entry from them
if [[ ${#ENTRIES[@]} -eq 0 && -n "${CUDA_SHORT:-}" && -n "${TORCH:-}" && -n "${PYTHON:-}" ]]; then
  ENTRIES+=("${CUDA_SHORT}:${TORCH}:${PYTHON}")
fi

# Default: full matrix
if [[ ${#ENTRIES[@]} -eq 0 ]]; then
  ENTRIES=("${ALL_ENTRIES[@]}")
fi

# ---------- conda init helper ----------
setup_conda() {
  if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found on PATH" >&2
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"

  # Ensure pip/requests use the system CA bundle (handles corporate proxies
  # that re-sign TLS with a custom root CA not in Python's certifi bundle).
  if [[ -z "${SSL_CERT_FILE:-}" ]]; then
    for ca in /etc/ssl/certs/ca-certificates.crt \
              /etc/pki/tls/certs/ca-bundle.crt \
              /etc/ssl/ca-bundle.pem; do
      if [[ -f "$ca" ]]; then
        export SSL_CERT_FILE="$ca"
        export REQUESTS_CA_BUNDLE="$ca"
        echo "Using system CA bundle: $ca"
        break
      fi
    done
  fi
}

# ---------- per-entry install & test (runs in a subshell) ----------
run_entry() {
  set -euo pipefail
  local cuda_short="$1" torch_ver="$2" py_ver="$3"
  local env_name="fusedops_test_cu${cuda_short}_torch${torch_ver}_py${py_ver}"
  local pkg_name="fireants-fused-ops-cu${cuda_short}"
  local pkg_spec="$pkg_name"
  if [[ -n "$VERSION" ]]; then
    pkg_spec="${pkg_name}==${VERSION}"
  fi

  echo ""
  echo "============================================================"
  echo "  CUDA_SHORT=$cuda_short  TORCH=$torch_ver  PYTHON=$py_ver"
  echo "  package: $pkg_spec  (from $PYPI_INDEX)"
  echo "  conda env: $env_name"
  echo "============================================================"

  echo ">>> Creating conda env (python=${py_ver})"
  conda create -n "$env_name" python="$py_ver" -y -q

  echo ">>> Activating env"
  conda activate "$env_name"

  echo ">>> Installing PyTorch ${torch_ver} (cu${cuda_short})"
  python -m pip install --no-cache-dir -q \
    torch=="${torch_ver}" \
    --index-url "https://download.pytorch.org/whl/cu${cuda_short}"

  echo ">>> Installing ${pkg_spec} from ${PYPI_INDEX}"
  # Use --no-deps to avoid the fused-ops 'torch>=2.1.0' requirement pulling
  # a different torch build (e.g. CPU-only from PyPI) over our cu-specific one.
  python -m pip install --no-cache-dir --no-deps \
    --index-url "$PYPI_INDEX" \
    --extra-index-url "https://pypi.org/simple/" \
    "$pkg_spec"

  # Install fireants as editable WITHOUT pulling deps — torch is already
  # installed at the exact version we need and SimpleITK==2.2.1 may not
  # exist for this Python.  Only the fireants source tree is needed for
  # the test imports.
  echo ">>> Installing fireants (local, no-deps)"
  python -m pip install --no-cache-dir -q --no-deps -e .

  # Install minimal test deps that aren't already present
  echo ">>> Installing test dependencies"
  python -m pip install --no-cache-dir -q pytest numpy scipy scikit-image \
    nibabel matplotlib tqdm pandas hydra-core SimpleITK 2>/dev/null \
    || python -m pip install --no-cache-dir -q pytest numpy scipy scikit-image \
         nibabel matplotlib tqdm pandas hydra-core

  echo ">>> Verifying import"
  python -c "import torch; print('torch', torch.__version__); import fireants_fused_ops; print('fireants_fused_ops loaded OK')"

  if [[ "$IMPORT_ONLY" == true ]]; then
    echo ">>> --import-only: skipping tests"
  else
    echo ">>> Running fused-ops tests"
    python -m pytest -v tests/test_fusedops*.py
  fi

  echo ">>> Entry cu${cuda_short}/torch${torch_ver}/py${py_ver} — PASS"
}

# ---------- main ----------
setup_conda

declare -A RESULTS
overall_rc=0

for entry in "${ENTRIES[@]}"; do
  IFS=: read -r cuda_short torch_ver py_ver <<< "$entry"
  env_name="fusedops_test_cu${cuda_short}_torch${torch_ver}_py${py_ver}"

  # Temporarily disable errexit so the outer script doesn't abort on failure,
  # but the *subshell* still has set -e active (re-enabled inside run_entry).
  set +e
  ( run_entry "$cuda_short" "$torch_ver" "$py_ver" )
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    RESULTS["$entry"]="PASS"
  else
    RESULTS["$entry"]="FAIL"
    overall_rc=1
  fi

  # Always clean up the conda env
  echo ">>> Cleaning up conda env: $env_name"
  conda env remove -n "$env_name" -y 2>/dev/null || true
done

# ---------- summary ----------
echo ""
echo "======================== SUMMARY ========================"
printf "%-8s  %-8s  %-6s  %s\n" "CUDA" "TORCH" "PYTHON" "RESULT"
printf "%-8s  %-8s  %-6s  %s\n" "------" "------" "------" "------"
for entry in "${ENTRIES[@]}"; do
  IFS=: read -r cuda_short torch_ver py_ver <<< "$entry"
  printf "%-8s  %-8s  %-6s  %s\n" "cu$cuda_short" "$torch_ver" "$py_ver" "${RESULTS[$entry]}"
done
echo "========================================================="

exit "$overall_rc"
