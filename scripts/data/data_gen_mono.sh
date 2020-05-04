CODE_HOME_DIR=/home/t-yirwan/data/codes/
FS_DIR=$CODE_HOME_DIR/MT-pytorch-mtl/Transformer/
export PYTHONPATH=$FS_DIR:$PYTHONPATH

DATA_HOME_DIR=~/data/data/wmt/
MTL_DIR=$DATA_HOME_DIR/wmt_mtl
MONO_DIR=$DATA_HOME_DIR/wmt_mono
MTL_LIB_DIR=$MTL_DIR/lib
MONO_LIB_DIR=$MONO_DIR/lib


DATA_GEN_BT()
{
    DATA_SETTING=${1? "DATA_SETTING: filt20m"}
    SPM_SETTING=${2? "SPM_SETTING: l10d02"}
    LANGS=${3? "SRC: de"}
    TGT=${4:-en}

    SPM_DIR=$MTL_LIB_DIR/spm_$SPM_SETTING
    TEXT_DIR=$MONO_LIB_DIR/text_bt_${DATA_SETTING}_${SPM_SETTING}
    BIN_DIR=$MONO_DIR/data_bt_${DATA_SETTING}_${SPM_SETTING}
    TMP_DIR=$BIN_DIR/tmp
    # rm -r $BIN_DIR
    mkdir -p $BIN_DIR $TMP_DIR

    # convert spm vocab to fairseq dict
    if [ ! -f $BIN_DIR/dict.src.txt ];
    then
        cp $SPM_DIR/*.model $BIN_DIR
        cp $SPM_DIR/*.vocab $BIN_DIR
        FVOCAB=$(ls $BIN_DIR/ | grep "spm[0-9a-z\_\.]*.vocab")
        echo "converting spm vocab: $FVOCAB"
        cut -f 1 $BIN_DIR/$FVOCAB | tail -n +4 | sed "s/$/ 100/g" > $BIN_DIR/dict.src.txt
    fi
    cp $BIN_DIR/dict.src.txt $BIN_DIR/dict.tgt.txt
    wc -l $BIN_DIR/spm*
    wc -l $BIN_DIR/dict*

    for SRC in $LANGS;
    do
        # (X', EN)
        echo -en "\nbuild $SRC-en\n"
        FSRC=$TEXT_DIR/mono_spm.en.trans.$SRC
        FTGT=$TEXT_DIR/mono_spm.en
        echo "SRC: $FSRC"
        echo "TGT: $FTGT"
        rm $TMP_DIR/*
        cp $FSRC $TMP_DIR/train_spm_$SRC$TGT.$SRC
        cp $FTGT $TMP_DIR/train_spm_$SRC$TGT.$TGT
        ls -lh $TMP_DIR/
        python $FS_DIR/preprocess.py \
            --task "translation" \
            --source-lang $SRC \
            --target-lang $TGT \
            --trainpref $TMP_DIR/train_spm_$SRC$TGT \
            --destdir $BIN_DIR \
            --padding-factor 1 \
            --workers 64 \
            --dataset-impl 'mmap' \
            --srcdict $BIN_DIR/dict.src.txt \
            --tgtdict $BIN_DIR/dict.tgt.txt

        # (EN', X)
        echo -en "\nbuild en-$SRC\n"
        FSRC=$TEXT_DIR/mono_spm.$SRC.trans.en
        FTGT=$TEXT_DIR/mono_spm.$SRC
        echo "SRC: $FSRC"
        echo "TGT: $FTGT"
        rm $TMP_DIR/*
        cp $FSRC $TMP_DIR/train_spm_$TGT$SRC.$TGT
        cp $FTGT $TMP_DIR/train_spm_$TGT$SRC.$SRC
        ls -lh $TMP_DIR/
        python $FS_DIR/preprocess.py \
            --task "translation" \
            --source-lang $TGT \
            --target-lang $SRC \
            --trainpref $TMP_DIR/train_spm_$TGT$SRC \
            --destdir $BIN_DIR \
            --padding-factor 1 \
            --workers 64 \
            --dataset-impl 'mmap' \
            --srcdict $BIN_DIR/dict.src.txt \
            --tgtdict $BIN_DIR/dict.tgt.txt
    done
    ls -lh $BIN_DIR
}

DATA_GEN_MONO()
{
    DATA_SETTING=${1? "DATA_SETTING: filt5m"}
    SPM_SETTING=${2? "SPM_SETTING: l07d01"}
    LANGS=${3? "LANGS: de en"}

    SPM_DIR=$MTL_LIB_DIR/spm_$SPM_SETTING
    TEXT_DIR=$MONO_LIB_DIR/text_mono_${DATA_SETTING}_${SPM_SETTING}
    BIN_DIR=$DATA_HOME_DIR/wmt_mono/data_mono_${DATA_SETTING}_${SPM_SETTING}
    mkdir -p $BIN_DIR

    # convert spm vocab to fairseq dict
    if [ ! -f $BIN_DIR/dict.src.txt ];
    then
        cp $SPM_DIR/*.model $BIN_DIR
        cp $SPM_DIR/*.vocab $BIN_DIR
        FVOCAB=$(ls $BIN_DIR/ | grep "spm[0-9a-z\_\.]*.vocab")
        echo "converting spm vocab: $FVOCAB"
        cut -f 1 $BIN_DIR/$FVOCAB | tail -n +4 | sed "s/$/ 100/g" > $BIN_DIR/dict.src.txt
    fi
    cp $BIN_DIR/dict.src.txt $BIN_DIR/dict.tgt.txt
    wc -l $BIN_DIR/spm*
    wc -l $BIN_DIR/dict*

    # build train
    for SRC in $LANGS;
    do
        python $FS_DIR/preprocess.py \
            --task "translation" \
            --source-lang $SRC \
            --trainpref $TEXT_DIR/mono_spm \
            --destdir $BIN_DIR \
            --padding-factor 1 \
            --workers 64 \
            --srcdict $BIN_DIR/dict.src.txt \
            --dataset-impl 'mmap' \
            --only-source
        for f in $(ls $BIN_DIR/train.*None.*); do mv $f ${f/$SRC-None.$SRC/$SRC}; done
    done
    ls -lh $BIN_DIR
}

# DATA_GEN_BT   "filt20m" "l10d02" "fr cs de fi lv et ro tr hi gu" "en"
DATA_GEN_MONO "filt50m" "l10d02" "en tr"