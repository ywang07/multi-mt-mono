echo "======================= GPU & CUDA Version Checks ========================"
nvidia-smi
cat /usr/local/cuda/version.txt
nvcc -V
free -h
echo "PHILLY_GPU_COUNT" $PHILLY_GPU_COUNT

echo "===================== Python & PyTorch Version Checks ===================="
python -V
python -c 'import torch; print(torch.__version__)'

echo "=============================== SETTINGS ================================="

DATA_DIR=$1
PROJ_NAME=$2
EXP_NAME=$3
ADDITIONAL_ARGS=$4

CODE_HOME_DIR=/mnt/default/codes
DATA_HOME_DIR=/mnt/default/data/wmt/wmt_mtl
FS_DIR=$CODE_HOME_DIR/MT-pytorch-mtl/Transformer
export PYTHONPATH=$FS_DIR:$PYTHONPATH

OUTPUT_DIR=/mnt/output/projects/$PROJ_NAME/$EXP_NAME/pt-results/models
mkdir -p $OUTPUT_DIR
if [ ! -s $OUTPUT_DIR/checkpoint_last.pt ];
then
    CKPT=$(ls $OUTPUT_DIR | grep "checkpoint[_0-9]*.pt" | sort -n | tail -1)
    if [ -f $OUTPUT_DIR/$CKPT ];
    then
        echo "set $CKPT as checkpoint_last.pt"
        cp $OUTPUT_DIR/$CKPT $OUTPUT_DIR/checkpoint_last.pt
    fi
fi

echo "PROJ_NAME:     $PROJ_NAME"
echo "EXP_NAME:      $EXP_NAME"
echo "DATA_DIR:      $DATA_DIR"
echo "MODEL_DIR:     $OUTPUT_DIR"
ls $OUTPUT_DIR

echo "============================= FairSeq Main ==============================="

python $FS_DIR/train.py $DATA_DIR \
        --save-dir $OUTPUT_DIR \
        $ADDITIONAL_ARGS \
        --share-all-embeddings \
        --fp16 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --lr 0.0005 \
        --warmup-updates 4000 \
        --warmup-init-lr 1e-07 \
        --min-lr 1e-09  \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --update-freq 16 \
        --max-update 400000 \
        --log-interval 100 \
        --no-progress-bar \
        2>&1 \
        | tee -a $OUTPUT_DIR/train.log