#!/bin/bash
# filepath: /home/files/git/Stylometrics/scripts/repair_structure.sh

# --- repair_structure.sh ---
# Attempts to fix deeply nested directory structures in the Stylometrics project.
# WARNING: Modifies the filesystem. BACK UP YOUR WORK or use Git. Run from project root.
# See: scripts/repair_structure.ApiNotes.md

set -e # Exit immediately if a command exits with a non-zero status.
# set -x # Uncomment for debugging - prints each command before execution

PROJECT_ROOT=$(pwd)
echo "Running structure repair in: ${PROJECT_ROOT}"
echo "WARNING: This script will move files and directories. Ensure you have a backup or use Git."
read -p "Press Enter to continue, or Ctrl+C to abort..."

# --- Target Structure Definition ---
TARGET_SRC_DIR="${PROJECT_ROOT}/src"
TARGET_EXAMPLES_DIR="${PROJECT_ROOT}/examples"
declare -a TARGET_SUBDIRS=("types" "utils" "matrix" "carriers")

# --- Create Target Directories ---
echo "Ensuring target directories exist..."
mkdir -p "${TARGET_EXAMPLES_DIR}"
echo "  Created/Ensured: ${TARGET_EXAMPLES_DIR}"
mkdir -p "${TARGET_SRC_DIR}"
echo "  Created/Ensured: ${TARGET_SRC_DIR}"
for subdir in "${TARGET_SUBDIRS[@]}"; do
    mkdir -p "${TARGET_SRC_DIR}/${subdir}"
    echo "  Created/Ensured: ${TARGET_SRC_DIR}/${subdir}"
done

# --- Identify and Move Misplaced Content ---
# This part makes assumptions based on the observed problematic path.
# It looks for the *deepest* occurrence of standard directory names and moves their *contents*.

echo "Searching for and moving misplaced content (using 'mv -n' for safety)..."

# Function to find the deepest directory matching a pattern and move its contents
move_deepest_contents() {
    local pattern_dir_name="$1" # e.g., "types"
    local target_dir="$2"       # e.g., "${TARGET_SRC_DIR}/types"
    local search_root="$3"      # e.g., "${TARGET_SRC_DIR}" or "${PROJECT_ROOT}"

    # Find directories matching the pattern, sort by depth (deepest first)
    # Using find + awk to calculate depth and sort
    local deepest_match
    deepest_match=$(find "${search_root}" -type d -name "${pattern_dir_name}" -mindepth 2 -print0 |
                    xargs -0 -I {} bash -c 'echo "$(echo "{}" | tr -cd "/" | wc -c) {}"' |
                    sort -nr | head -n 1 | cut -d' ' -f2-)

    if [[ -n "${deepest_match}" && -d "${deepest_match}" && "${deepest_match}" != "${target_dir}" ]]; then
        echo "  Found potentially misplaced content in: ${deepest_match}"
        # Check if source is not empty and target exists
        if [ "$(ls -A "${deepest_match}")" ] && [ -d "${target_dir}" ]; then
            echo "    Moving contents of '${deepest_match}'/* to '${target_dir}/'..."
            # Move contents (*). Use -n to avoid overwriting existing files in the target.
            # Consider removing -n carefully if overwrites are intended and understood.
            mv -n "${deepest_match}"/* "${target_dir}/" || echo "    Warning: 'mv -n' may have skipped some files due to existing names in target."

            # Optional: Attempt to remove the now-empty source directory.
            # This might fail if intermediate parent directories also became empty and need removal.
            # Use with caution.
            # rmdir "${deepest_match}" 2>/dev/null || echo "    Info: Could not remove source directory '${deepest_match}' (might not be empty or intermediate dirs exist)."
            echo "    (Consider manually cleaning up empty parent directories like '${deepest_match}' and its parents if the move was successful)"
        else
             echo "    Skipping move: Source '${deepest_match}' is empty or target '${target_dir}' does not exist."
        fi
    else
        echo "  No misplaced '${pattern_dir_name}' directories found needing move under '${search_root}'."
    fi
}

# Move contents for src subdirectories
for subdir in "${TARGET_SUBDIRS[@]}"; do
    move_deepest_contents "${subdir}" "${TARGET_SRC_DIR}/${subdir}" "${TARGET_SRC_DIR}"
done

# Move contents for examples
move_deepest_contents "examples" "${TARGET_EXAMPLES_DIR}" "${PROJECT_ROOT}"


# --- Final Check/Cleanup Suggestions ---
echo ""
echo "--- Repair Script Finished ---"
echo "Review the output above for any warnings or errors."
echo "Check the following directories for correct content:"
echo "  ${TARGET_EXAMPLES_DIR}"
for subdir in "${TARGET_SUBDIRS[@]}"; do
    echo "  ${TARGET_SRC_DIR}/${subdir}"
done
echo "Manually inspect and remove any remaining empty nested directories if the moves were successful."
echo "Use 'git status' to review changes before committing."
