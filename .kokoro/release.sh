#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Release script for the Google Antigravity SDK.
#
# Builds platform-specific wheels containing the pre-compiled Go
# localharness binary and uploads them to the OSS Exit Gate for
# distribution.
#
# This script works in two modes:
#   1. Kokoro: Runs inside a Kokoro release job. Binaries are fetched
#      from MPM via Kokoro's fetch_mpm (pre-populated by Rapid).
#   2. Local: Run from a Copybara export directory after manually
#      placing binaries under .kokoro/binaries/<platform>/localharness.
#
# Environment variables:
#   VERSION         - SDK version (default: auto-read from pyproject.toml).
#   PUBLISH         - If set to "true", uploads a manifest to the OSS Exit
#                     Gate GCS bucket after the wheel upload, triggering
#                     promotion to public PyPI.  Without this, wheels are
#                     staged in Artifact Registry but NOT made public.
#
# Usage (local, after Copybara export):
#   # Place binary(ies) under .kokoro/binaries/<platform>/localharness,
#   # then run:
#   VERSION=0.1.1 .kokoro/release.sh
#
# Usage (single platform, local shortcut):
#   mkdir -p .kokoro/binaries/linux-x86_64
#   cp /path/to/localharness .kokoro/binaries/linux-x86_64/localharness
#   .kokoro/release.sh

set -eo pipefail

# --- Resolve SCRIPT_DIR before cd so the relative BASH_SOURCE path is
# resolved against the original working directory (the Kokoro workspace root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Determine project root ---
if [[ -n "${KOKORO_ARTIFACTS_DIR}" ]]; then
  cd "${KOKORO_ARTIFACTS_DIR}/git/antigravity-sdk-py"
  # Avoid fatal: detected dubious ownership errors in the Kokoro RBE sandbox environment
  git config --global --add safe.directory "$(pwd)"
fi

# If a specific GoB commit was requested (via GOB_COMMIT build param),
# check it out to pin the Python source to a specific Piper CL.
# Kokoro exposes build_params as environment variables.
if [[ -n "${GOB_COMMIT:-}" ]]; then
  echo "--- Pinning SDK source to GoB commit: ${GOB_COMMIT} ---"
  git checkout "${GOB_COMMIT}"
  git checkout HEAD -- .kokoro/
fi

PROJECT_DIR="$(pwd)"

# --- Python 3.10 via pyenv (pre-installed on the Kokoro image) ---
echo "--- Setting up Python 3.10 ---"
eval "$(pyenv init -)"
pyenv install -s 3.10
pyenv global 3.10
python3 --version

# --- Read version from pyproject.toml if not set ---
if [[ -n "${PUBLISH_PREBUILT_VERSION:-}" ]]; then
  VERSION="${PUBLISH_PREBUILT_VERSION}"
  echo "=== RUNNING IN PUBLISH-ONLY MODE FOR VERSION v${VERSION} ==="

  DIST_DIR="dist"
  rm -rf "${DIST_DIR}"
  mkdir -p "${DIST_DIR}"

  GCS_SOURCE="gs://agy-sdk-wheels/v${VERSION}"
  echo "--- Downloading pre-built wheels from ${GCS_SOURCE} ---"

  # Impersonate the agy-sdk-stager service account keylessly using Kokoro's ambient credentials
  echo "--- Activating agy-sdk-stager impersonation ---"

  # Isolate the gcloud config directory to prevent parallel SQLite token database locks
  GCLOUD_TEMP_CONFIG=$(mktemp -d)
  (
    export CLOUDSDK_CONFIG="${GCLOUD_TEMP_CONFIG}"
    gcloud config set auth/impersonate_service_account agy-sdk-stager@agy-sdk.iam.gserviceaccount.com
    gcloud storage cp "${GCS_SOURCE}"/*.whl "${DIST_DIR}/"
  )
  rm -rf "${GCLOUD_TEMP_CONFIG}"
else
  if [[ -z "${VERSION}" ]]; then
    VERSION=$(sed -n '/^\[project\]/,/^\[/p' pyproject.toml | grep -E '^version\s*=' | cut -d'"' -f2)
  fi
  echo "=== Google Antigravity SDK Release v${VERSION} ==="
fi



# Install build/release tools with hash verification.
# See go/pip-install-remediation.
python3 -m pip install \
  --require-hashes \
  -r "${SCRIPT_DIR}/requirements-release.txt"

if [[ -z "${PUBLISH_PREBUILT_VERSION:-}" ]]; then
  DIST_DIR="dist"
  rm -rf "${DIST_DIR}"
  mkdir -p "${DIST_DIR}"
fi

# --- Platform definitions ---
declare -A PLATFORM_TAGS=(
  ["linux-x86_64"]="manylinux_2_17_x86_64"
  ["linux-arm64"]="manylinux_2_17_aarch64"
  ["darwin-arm64"]="macosx_11_0_arm64"
  ["windows-x86_64"]="win_amd64"
  ["windows-arm64"]="win_arm64"
)

if [[ -z "${PUBLISH_PREBUILT_VERSION:-}" ]]; then
  BINARY_NAME="localharness"
  BIN_DEST="google/antigravity/bin"
  BINARIES_DIR=".kokoro/binaries"

  # --- MPM directory mapping (populated by Kokoro fetch_mpm) ---
  MPM_DIR="${KOKORO_ARTIFACTS_DIR:-}/mpm"
  declare -A MPM_DIRS=(
    ["linux-x86_64"]="localharness_linux_x86_64"
    ["linux-arm64"]="localharness_linux_arm64"
    ["darwin-arm64"]="localharness_darwin_arm64"
    ["windows-x86_64"]="localharness_windows_x86_64"
    ["windows-arm64"]="localharness_windows_arm64"
  )

  # --- Fetch binaries from MPM or local ---
  for PLATFORM in "${!PLATFORM_TAGS[@]}"; do
    LOCAL_BIN="${BINARIES_DIR}/${PLATFORM}/${BINARY_NAME}"
    if [[ ! -f "${LOCAL_BIN}" ]]; then
      MPM_SUBDIR="${MPM_DIRS[$PLATFORM]:-}"
      MPM_BIN="${MPM_DIR}/${MPM_SUBDIR}/localharness_external"
      if [[ -n "${MPM_SUBDIR}" && -f "${MPM_BIN}" ]]; then
        echo "--- Copying ${PLATFORM} binary from MPM ---"
        mkdir -p "${BINARIES_DIR}/${PLATFORM}"
        cp "${MPM_BIN}" "${LOCAL_BIN}"
        chmod +x "${LOCAL_BIN}"
      else
        if [[ -n "${KOKORO_ARTIFACTS_DIR}" ]]; then
          echo "ERROR: No binary for ${PLATFORM} (looked in ${MPM_BIN})."
          echo "In a release job, all platform binaries must be available."
          exit 1
        fi
        echo "WARNING: No binary for ${PLATFORM} (looked in ${MPM_BIN}), skipping."
        continue
      fi
    fi
  done

  # --- Build platform-specific wheels ---
  BUILT_ANY=false

  for PLATFORM in "${!PLATFORM_TAGS[@]}"; do
    WHEEL_PLAT="${PLATFORM_TAGS[$PLATFORM]}"
    LOCAL_BIN="${BINARIES_DIR}/${PLATFORM}/${BINARY_NAME}"

    if [[ ! -f "${LOCAL_BIN}" ]]; then
      echo "--- Skipping ${PLATFORM}: no binary available ---"
      continue
    fi

    CUR_BIN_NAME="${BINARY_NAME}"
    if [[ "${PLATFORM}" == windows-* ]]; then
      CUR_BIN_NAME="${BINARY_NAME}.exe"
    fi

    echo "--- Building wheel for ${PLATFORM} (${WHEEL_PLAT}) ---"

    # Place the binary into the package namespace.
    mkdir -p "${BIN_DEST}"
    cp "${LOCAL_BIN}" "${BIN_DEST}/${CUR_BIN_NAME}"
    chmod +x "${BIN_DEST}/${CUR_BIN_NAME}"

    # Ensure __init__.py exists for the bin subpackage so setuptools
    # discovers it via package-data.
    touch "${BIN_DEST}/__init__.py"

    # Clean setuptools build directory to prevent binary accumulation across platforms.
    rm -rf build/
    # Build the wheel, then re-tag with the correct platform.
    # --no-isolation: use the setuptools/wheel already installed via
    # requirements-release.txt rather than creating a fresh venv that
    # downloads from PyPI (which can time out and bypasses hash verification).
    python3 -m build --wheel --no-isolation --outdir "${DIST_DIR}"
    python3 -m wheel tags \
      --platform-tag="${WHEEL_PLAT}" \
      --remove \
      "${DIST_DIR}"/*-py3-none-any.whl

    echo "  -> $(ls -1 "${DIST_DIR}"/*"${WHEEL_PLAT}"*.whl 2>/dev/null | tail -1)"

    # Clean the binary for the next platform iteration.
    rm -rf "${BIN_DEST}"
    BUILT_ANY=true
  done

  if [[ "${BUILT_ANY}" != "true" ]]; then
    echo "ERROR: No wheels were built. Ensure binaries are available."
    exit 1
  fi

  echo ""
  echo "--- Built wheels ---"
  ls -lh "${DIST_DIR}/"
fi

# ---------------------------------------------------------------------------
# Prepublish Staging Flow (Default Mode)
# ---------------------------------------------------------------------------
# If PUBLISH is not "true" and we are not running in publish-prebuilt mode,
# stage the wheels internally to GCS, then exit.
if [[ "${PUBLISH:-}" != "true" && -z "${PUBLISH_PREBUILT_VERSION:-}" ]]; then
  echo ""
  echo "=== PREPUBLISH: Staging wheels internally (Default Mode) ==="

  # 1. Upload wheels to GCS bucket
  GCS_DEST="gs://agy-sdk-wheels/v${VERSION}"
  echo "--- Uploading wheels to GCS staging: ${GCS_DEST}/ ---"
  
  # Impersonate the agy-sdk-stager service account keylessly using Kokoro's ambient credentials
  echo "--- Activating agy-sdk-stager impersonation ---"
  gcloud config set auth/impersonate_service_account agy-sdk-stager@agy-sdk.iam.gserviceaccount.com
  
  gcloud storage cp "${DIST_DIR}"/*.whl "${GCS_DEST}/"
  
  # Unset stager impersonation to return to default credentials
  gcloud config unset auth/impersonate_service_account

  echo "=== Prepublish staging complete ==="
  echo ""
  echo "  Staged wheels: ${GCS_DEST}/"
  exit 0
fi

# ---------------------------------------------------------------------------
# Publish to PyPI (Only when publishing or promoting)
# ---------------------------------------------------------------------------
if [[ "${PUBLISH:-}" == "true" || -n "${PUBLISH_PREBUILT_VERSION:-}" ]]; then
  # 1. Twine Upload to the secure OSS Exit Gate
  REPO_URL="https://us-python.pkg.dev/oss-exit-gate-prod/google-antigravity--pypi/"
  echo ""
  echo "--- Validating wheels ---"
  twine check "${DIST_DIR}"/*
  echo ""
  echo "--- Uploading to OSS Exit Gate (${REPO_URL}) ---"
  twine upload \
    --repository-url "${REPO_URL}" \
    --verbose \
    "${DIST_DIR}"/*

  # 2. Manifest upload — triggers promotion from AR staging to public PyPI.
  # The OSS Exit Gate uses a GCS manifest as the "publish now" signal.
  # Uploading this file triggers the Exit Gate to verify and publish all
  # staged artifacts to pypi.org. See go/oss-exit-gate-release-python.
  EG_GCS_BUCKET="gs://oss-exit-gate-prod-projects-bucket/google-antigravity/pypi/manifests"

  echo ""
  echo "--- Publishing to PyPI: uploading manifest to OSS Exit Gate ---"
  MANIFEST_FILE="manifest.json"
  echo '{ "publish_all": true }' > "${MANIFEST_FILE}"
  MANIFEST_NAME="manifest-v${VERSION}-$(date -u +%Y%m%d-%H%M%S).json"
  gcloud storage cp "${MANIFEST_FILE}" "${EG_GCS_BUCKET}/${MANIFEST_NAME}"
  rm -f "${MANIFEST_FILE}"

  echo "  Manifest uploaded: ${EG_GCS_BUCKET}/${MANIFEST_NAME}"
  echo "  The OSS Exit Gate will now verify and publish to pypi.org."
  echo "  Monitor progress at: http://go/spng2?q=PROJECT%3Agoogle-antigravity%2Fpypi"
  echo ""
  echo "--- Release v${VERSION} published ---"
else
  echo ""
  echo "--- Staging verified. Manifest upload & Exit Gate skipped (PUBLISH is false) ---"
  echo ""
fi
