You must already have folds.json in data folder (see other repo).

Prepare data::

    python -m fashion.data

Train::

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
        python tools/train.py fashion/configs/htc_r50_fpn_20e.py \
        --work_dir _runs/example

