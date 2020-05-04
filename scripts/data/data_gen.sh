CODE_HOME_DIR=/home/t-yirwan/data/codes/
FS_DIR=$CODE_HOME_DIR/MT-pytorch/Transformer/
export PYTHONPATH=$FS_DIR:$PYTHONPATH

DATA_GEN()
{
  SRC=${1? SRC: de}
  TGT=${2? TGT: en}
  YEAR=${3? YEAR: 19}
  SETTING=${4? SETTING: filt}
  FPREFIX=${5? FPREFIX: filt_spm}

  DATA_DIR=~/data/data/wmt/wmt${YEAR}_$SRC$TGT/
  TEXT_DIR=$DATA_DIR/lib/text_$SETTING
  BIN_DIR=$DATA_DIR/data_$SETTING

  rm -r $BIN_DIR
  mkdir -p $BIN_DIR

  # convert spm vocab to fairseq dict
  cp $TEXT_DIR/*.model $BIN_DIR
  cp $TEXT_DIR/*.vocab $BIN_DIR
  FVOCAB=$(ls $BIN_DIR/ | grep "spm[0-9a-z\_\.]*.vocab")
  echo "converting spm vocab: $FVOCAB"
  cut -f 1 $BIN_DIR/$FVOCAB | tail -n +4 | sed "s/$/ 100/g" > $BIN_DIR/dict.$TGT.txt
  cp $BIN_DIR/dict.$TGT.txt $BIN_DIR/dict.$SRC.txt
  wc -l $BIN_DIR/spm*
  wc -l $BIN_DIR/dict*

  python -V
  python -c 'import torch; print(torch.__version__)'
  python $FS_DIR/preprocess.py \
        --task "translation" \
        --source-lang $SRC \
        --target-lang $TGT \
        --trainpref $TEXT_DIR/train_spm \
        --validpref $TEXT_DIR/dev_${SRC}${TGT}_spm \
        --destdir $BIN_DIR \
        --padding-factor 1 \
        --workers 32 \
        --srcdict $BIN_DIR/dict.$SRC.txt \
        --tgtdict $BIN_DIR/dict.$TGT.txt
  ls -lh $BIN_DIR
}


SRC=${1? SRC: de}
TGT=${2? TGT: en}
YEAR=${3? YEAR: 19}
SETTING=${4? SETTING: filt}

DATA_GEN $SRC $TGT $YEAR $SETTING "${SETTING}_spm"


