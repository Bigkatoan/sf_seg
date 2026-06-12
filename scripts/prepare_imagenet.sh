#!/bin/bash
# Gắn ImageNet đã tải (kagglehub cache) vào data/imagenet/ cho ImageFolder.
#   - train: symlink thư mục (đã đúng dạng <wnid>/, không copy 150GB)
#   - val:   gom flat → <wnid>/ bằng symlink (theo LOC_val_solution.csv)
#
# Usage:
#   ./scripts/prepare_imagenet.sh [KAGGLE_PATH]
# Nếu bỏ trống, tự dò ~/.cache/kagglehub/competitions/imagenet-*/

set -e
source venv/bin/activate

KP="${1:-}"
if [ -z "$KP" ]; then
    KP=$(find ~/.cache/kagglehub/competitions -maxdepth 3 -type d \
         -name 'CLS-LOC' 2>/dev/null | head -1 | xargs -r dirname | xargs -r dirname | xargs -r dirname)
    # KP = thư mục chứa ILSVRC/ và LOC_val_solution.csv
    [ -z "$KP" ] && KP=$(find ~/.cache/kagglehub/competitions -maxdepth 2 -type d \
         -name 'imagenet-object-localization-challenge*' 2>/dev/null | head -1)
fi

CLS="$KP/ILSVRC/Data/CLS-LOC"
SOL="$KP/LOC_val_solution.csv"
echo "Kaggle path : $KP"
echo "train dir   : $CLS/train"
echo "val dir     : $CLS/val"
echo "solution    : $SOL"

if [ ! -d "$CLS/train" ] || [ ! -f "$SOL" ]; then
    echo "LỖI: không thấy $CLS/train hoặc $SOL — kiểm tra lại đường dẫn (truyền tay: ./scripts/prepare_imagenet.sh /path)"
    exit 1
fi

mkdir -p data/imagenet

# train: symlink toàn thư mục (đã <wnid>/*.JPEG)
ln -sfn "$(realpath "$CLS/train")" data/imagenet/train
echo "train: $(ls data/imagenet/train | wc -l) wnid folders (symlinked)"

# val: gom theo wnid bằng symlink
python -m scripts.restructure_in1k_val \
    --val-dir "$CLS/val" --solution "$SOL" \
    --out data/imagenet/val --symlink

echo
echo "Done. Kiểm tra nhanh:"
echo "  train classes: $(ls data/imagenet/train | wc -l)  (cần 1000)"
echo "  val classes  : $(ls data/imagenet/val | wc -l)  (cần 1000)"
echo
echo "Train: ./train_imagenet.sh"
