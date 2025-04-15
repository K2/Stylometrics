#!/bin/bash
# filepath: /home/files/git/Stylometrics/scripts/cleanup_empty_dirs.sh

# --- cleanup_empty_dirs.sh ---
# Removes empty directories within the 'src/' directory.
# Intended for use AFTER 'repair_structure.sh' to clean up leftovers.
# WARNING: Modifies the filesystem. BACK UP YOUR WORK or use Git. Run from project root.
# See: scripts/cleanup_empty_dirs.ApiNotes.md

set -e # Exit immediately if a command exits with a non-zero status.

PROJECT_ROOT=$(pwd)
SRC_DIR="${PROJECT_ROOT}/src"

echo "Running empty directory cleanup within: ${SRC_DIR}"

if [ ! -d "${SRC_DIR}" ]; then
    echo "Error: Directory '${SRC_DIR}' not found. Exiting."
    exit 1
fi

echo "This script will find and attempt to remove empty directories inside '${SRC_DIR}'."
echo "It uses 'find ... -type d -empty -delete', operating from deepest directories upwards."
echo "WARNING: Review the command and ensure you understand its implications."
echo "Make sure 'scripts/repair_structure.sh' has been run successfully first."

# List empty directories first for review (optional but recommended)
echo ""
echo "Empty directories found within '${SRC_DIR}' (candidates for deletion):"
# Use -mindepth 1 to avoid listing src/ itself if it were empty
find "${SRC_DIR}" -mindepth 1 -type d -empty -print || echo "(No empty directories found)"
echo ""

read -p "Proceed with deleting these empty directories? (y/N): " confirm
confirm=${confirm:-N} # Default to No

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 0
fi

echo "Attempting to delete empty directories..."

# Use find with -delete. This is generally safe as it only deletes empty directories
# and processes them depth-first, so it removes nested empty dirs correctly.
find "${SRC_DIR}" -mindepth 1 -type d -empty -delete

echo ""
echo "--- Cleanup Script Finished ---"
echo "Empty directory removal process completed."
echo "Use 'git status' to review changes (deleted directories might not show explicitly, but check containing directories)."
echo "Verify that no essential directories were accidentally removed."
