echo "======================= GPU & CUDA Version Checks ========================"
nvidia-smi
cat /usr/local/cuda/version.txt
nvcc -V
echo "PHILLY_GPU_COUNT" $PHILLY_GPU_COUNT

echo "===================== Python & PyTorch Version Checks ===================="
python -V
python -c 'import torch; print(torch.__version__)'

echo "=============================== SETTINGS ================================="

declare -A TEST_SETS=(
    ['fr']="wmt13 wmt14 wmt15"
    ['de']="wmt16 wmt17 wmt18 wmt19"
    ['fi']="wmt16 wmt17 wmt18 wmt19"
    ['cs']="wmt16 wmt17 wmt18 wmt19"
    ['et']="dev18 wmt18"
    ['tr']="wmt16 wmt17 wmt18"
    ['lv']="dev17 wmt17"
    ['ro']="dev16 wmt16"
    ['hi']="dev14 wmt14"
    ['gu']="dev19 wmt19"
)

DATA_DIR=$1
PROJ_NAME=$2        # wmt_mtl
EXP_NAME=$3         # 001_xxen_offupsp_t5_s13m
SRC_LANGS=$4        # "de fi cs"
TGT=$5              # en
REV=$6              # true/false
ADDITIONAL_ARGS=$7
LAST_N_CKPT=${8:-5}

# code path
CODE_HOME_DIR=/mnt/default/codes
FS_DIR=$CODE_HOME_DIR/MT-pytorch-mtl/Transformer
SACRE_BLEU=$CODE_HOME_DIR/sacreBLEU
export PYTHONPATH=$FS_DIR:$PYTHONPATH

# model setting
BEAM_SIZE=5
LPEN=1.0
BATCH_SIZE=200

if [[ $SRC_LANGS == "all7" ]]; then SRC_LANGS="de fi cs et tr lv ro"; fi
if [[ $SRC_LANGS == "all10" ]]; then SRC_LANGS="fr de fi cs et tr lv ro hi gu"; fi

# data path
DATA_HOME_DIR=/mnt/default/data/wmt/
F=$(ls $DATA_DIR | grep "spm[a-z0-9\_]*.model")
SPM_MODEL=$DATA_DIR/$F

# result path
ALL_RESULTS=/mnt/output/projects/$PROJ_NAME/ALL_BLEU/${EXP_NAME}
RESULTS_DIR=/mnt/output/projects/$PROJ_NAME/$EXP_NAME/pt-results/
TRAIN_DIR=$RESULTS_DIR/models
TRANS_DIR=$RESULTS_DIR/trans
ALL_BLEU_DIR=$RESULTS_DIR/ALL_BLEU
TMP_DIR=/tmp
mkdir -p $TMP_DIR $TRANS_DIR $ALL_BLEU_DIR $ALL_RESULTS

ALL_CKPT=$(ls -rv $TRAIN_DIR | grep "checkpoint" | head -$LAST_N_CKPT)
# ALL_CKPT=$(ls -rv $TRAIN_DIR | grep "checkpoint[0-9]*.pt" | head -$LAST_N_CKPT)

echo "PROJ_NAME:  $PROJ_NAME"
echo "EXP_NAME:   $EXP_NAME"
echo "DATA_DIR:   $DATA_DIR"
echo "SPM_MODEL:  $SPM_MODEL"
echo "TRAIN_DIR:  $TRAIN_DIR"
echo "SRC_LANGS:  $SRC_LANGS"
echo "ALL_CKPT:   $ALL_CKPT"

echo "============================= FairSeq Main ==============================="

for CKPT in $ALL_CKPT;
do
    CKPT_ID=$( cut -d "." -f 1 <<< $CKPT )
    TMP_TEST_DIR=$TMP_DIR/$EXP_NAME-$CKPT_ID-beam$BEAM_SIZE-lpen$LPEN
    mkdir -p $TMP_TEST_DIR
    cp $TRAIN_DIR/$CKPT $TMP_TEST_DIR

    for SRC in $SRC_LANGS;
    do
        SRCLAN=$SRC; TGTLAN=$TGT; LAN_PAIR=$SRC$TGT
        if [[ $REV == "true" ]]; then SRCLAN=$TGT; TGTLAN=$SRC; LAN_PAIR=$TGT$SRC; fi
        echo "LAN_PAIR: $LAN_PAIR, SRCLAN: $SRCLAN, TGTLAN: $TGTLAN"
        YEARS=${TEST_SETS[$SRC]}
        echo "YEARS: $YEARS"

        TEST_DATA_DIR=$DATA_HOME_DIR/wmt_${SRC}${TGT}/test_set
        TEST_RESULT_DIR=$TRANS_DIR/$LAN_PAIR-$CKPT_ID-beam$BEAM_SIZE-lpen$LPEN
        mkdir -p $TEST_RESULT_DIR
        echo "TEST_DATA_DIR: $TEST_DATA_DIR"
        echo "TEST_RESULT_DIR: $TEST_RESULT_DIR"

        for YEAR in $YEARS;
        do
            if [ -s $TEST_RESULT_DIR/BLEU.$LAN_PAIR.$YEAR.$CKPT_ID.txt ] && [[ -n $(echo $CKPT_ID | grep -o "[0-9]") ]];
            then
                echo "skip $CKPT ($LAN_PAIR, $YEAR): already tested"
            else
                echo "testing $CKPT ($LAN_PAIR, $YEAR)"
                ALL_BLEU_FILE=$ALL_BLEU_DIR/ALL_BLEU-$LAN_PAIR-$YEAR-beam$BEAM_SIZE-lpen$LPEN.txt
                FSRC=$YEAR.$LAN_PAIR.$SRCLAN
                FTGT=$YEAR.$LAN_PAIR.$TGTLAN
                FOUT=$LAN_PAIR.$YEAR.$CKPT_ID
                FBLEU=BLEU.$FOUT.txt
                cp $TEST_DATA_DIR/$FSRC $TMP_TEST_DIR
                cp $TEST_DATA_DIR/$FTGT $TMP_TEST_DIR
                # ls -lh $TMP_TEST_DIR

                # apply SPM
                python $FS_DIR/scripts/spm_encode.py --model $SPM_MODEL --inputs $TMP_TEST_DIR/$FSRC --outputs $TMP_TEST_DIR/$FSRC.spm

                # decode
                cat $TMP_TEST_DIR/$FSRC.spm | \
                python $FS_DIR/interactive.py $DATA_DIR \
                    --path $TMP_TEST_DIR/$CKPT \
                    $ADDITIONAL_ARGS \
                    --source-lang $SRCLAN \
                    --target-lang $TGTLAN \
                    --buffer-size 1024 \
                    --batch-size $BATCH_SIZE \
                    --beam $BEAM_SIZE \
                    --lenpen $LPEN \
                    --remove-bpe=sentencepiece \
                    --no-progress-bar \
                    > $TMP_TEST_DIR/$FOUT
                cat $TMP_TEST_DIR/$FOUT | grep -P "^H" | cut -f 3- > $TMP_TEST_DIR/$FOUT.out
                cp $TMP_TEST_DIR/$FOUT     $TEST_RESULT_DIR
                cp $TMP_TEST_DIR/$FOUT.out $TEST_RESULT_DIR
                # ls -lh $TMP_TEST_DIR

                cat $TMP_TEST_DIR/$FOUT.out | python $SACRE_BLEU/sacrebleu.py $TMP_TEST_DIR/$FTGT | grep "BLEU" >> $TMP_TEST_DIR/$FBLEU
                echo -en "$CKPT_ID\t"    >> $ALL_BLEU_FILE
                cat $TMP_TEST_DIR/$FBLEU >> $ALL_BLEU_FILE
                cat $TMP_TEST_DIR/$FBLEU
                cp  $TMP_TEST_DIR/$FBLEU $TEST_RESULT_DIR
            fi
        done
        cp $ALL_BLEU_DIR/ALL_BLEU-$LAN_PAIR-* $ALL_RESULTS/
    done
    rm -r $TMP_TEST_DIR
done