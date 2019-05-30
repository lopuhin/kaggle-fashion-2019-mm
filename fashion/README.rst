You must already have folds.json in data folder (see other repo).

Prepare data::

    python -m fashion.data

Convert model trained on COCO::

    python -m fashion.convert_pretrained \
        htc_r50_fpn_20e_20190408-c03b7015.pth \
        htc_r50_fpn_20e_20190408-c03b7015_converted.pth

Train::

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
        python tools/train.py fashion/configs/htc_r50_fpn_20e.py \
        --resume_from htc_r50_fpn_20e_20190408-c03b7015_converted.pth
        --work_dir _runs/example

