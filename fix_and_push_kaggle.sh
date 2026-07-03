#!/usr/bin/env bash
set -e

BRANCH="claude/bms-battery-management-y26x71"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== BMS Kaggle Fix & Push ==="
echo "Repo: $REPO_DIR"

cd "$REPO_DIR"

rm -f .git/HEAD.lock .git/index.lock .git/refs/heads/*.lock 2>/dev/null || true

git fetch origin

git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"

mkdir -p data/kaggle

SEARCH_DIRS=(
    "$REPO_DIR/data/kaggle"
    "$HOME/data/kaggle"
    "$HOME/Downloads"
    "$HOME"
)

DATASETS=(
    "nasa:nasa"
    "degradation:degradation"
    "ev_charging:ev_charging"
    "rul:rul"
    "bms_v21:bms_v21"
    "dist_bms:dist_bms"
)

copy_if_found() {
    local src_name="$1"
    local dst_name="$2"

    for base in "${SEARCH_DIRS[@]}"; do
        if [ -d "$base/$src_name" ] && [ "$base/$src_name" != "$REPO_DIR/data/kaggle/$dst_name" ]; then
            echo "  Found $src_name at $base/$src_name"
            mkdir -p "data/kaggle/$dst_name"
            cp -rn "$base/$src_name/." "data/kaggle/$dst_name/" 2>/dev/null || true
            return 0
        fi
    done

    if find "$HOME" -maxdepth 5 -name "Battery_RUL.csv" 2>/dev/null | grep -q .; then
        CSV_PARENT=$(find "$HOME" -maxdepth 5 -name "Battery_RUL.csv" 2>/dev/null | head -1 | xargs dirname)
        KAGGLE_PARENT=$(dirname "$CSV_PARENT")
        if [ -d "$KAGGLE_PARENT/$src_name" ]; then
            echo "  Found $src_name at $KAGGLE_PARENT/$src_name"
            mkdir -p "data/kaggle/$dst_name"
            cp -rn "$KAGGLE_PARENT/$src_name/." "data/kaggle/$dst_name/" 2>/dev/null || true
            return 0
        fi
    fi

    return 1
}

for pair in "${DATASETS[@]}"; do
    src="${pair%%:*}"
    dst="${pair##*:}"
    copy_if_found "$src" "$dst" || echo "  WARNING: $src not found — will use synthetic data"
done

NOTEBOOK=$(find "$HOME" -maxdepth 5 -name "nasa-battery-life-prediction-dataset-cleaning.ipynb" 2>/dev/null | head -1)
if [ -n "$NOTEBOOK" ]; then
    echo "  Found notebook: $NOTEBOOK"
    cp "$NOTEBOOK" "$REPO_DIR/"
fi

echo ""
echo "Files found in data/kaggle/:"
find data/kaggle -name "*.csv" | while read f; do
    du -sh "$f"
done

CSV_COUNT=$(find data/kaggle -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')
if [ "$CSV_COUNT" -eq 0 ]; then
    echo ""
    echo "ERROR: No CSV files found anywhere. Run the download first:"
    echo "  /opt/homebrew/bin/kaggle datasets download patrickfleith/nasa-battery-dataset -p data/kaggle/nasa --unzip"
    exit 1
fi

git add data/kaggle/
git add nasa-battery-life-prediction-dataset-cleaning.ipynb 2>/dev/null || true

if git diff --cached --quiet; then
    echo "Nothing new to commit — already up to date."
else
    git commit -m "Add real Kaggle battery datasets (NASA, RUL, degradation, EV charging, BMS)"
fi

git push origin "$BRANCH"

echo ""
echo "=== Done! Data pushed to $BRANCH ==="
