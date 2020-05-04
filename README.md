# Introduction 
Multitask training for Multilingual Neural Machine Translation

#### Supported features:
* Baseline: Multilingual NMT system 
* Multitask objective: masked language model
* Multitask objective: denoising auto-encoder
* Data scheduling: dynamic temperature for data sampling
* Task scheduling: dynamic multitask weights and noising ratio

#### Table of Contents

* [Usage](#usage): example training / test commands
* [Data](#data): instructions on data construction and organization
* [Tasks](#tasks): description of self-defined tasks
* [Named Arguments](#named_args): full list of self-defined arguments
* [Scripts](#scripts): example scripts for training, test and data construction
* [Codes](#codes): list of files with major modifications

# <a name="usage"> Usage </a>

### MultiNMT Baseline

Training (online sampling with T=5)
```
python $FS_DIR/train.py $DATA_DIR --save-dir $OUTPUT_DIR \
    --arch transformer_vaswani_wmt_en_de_big \
    --task translation_mtl_ols \
    --lang-pairs "fr-en,de-en" \
    --language-sample-temperature 5.0 \
    --language-temperature-scheduler "static" \
    --dataset-impl mmap --share-all-embeddings --fp16 --ddp-backend=no_c10d \
    --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0 \
    --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --warmup-init-lr 1e-07 --min-lr 1e-09  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --max-update 50000 --update-freq 16 --log-interval 100 \
    --save-interval-updates 4000 --no-progress-bar
```

Test (pair FR-EN)
```
cat $FSRC.spm | python $FS_DIR/interactive.py $DATA_DIR --path $CKPT \
    --task translation_mtl_ols \
    --lang-pairs "fr-en,de-en" \
    --source-lang fr --target-lang en \
    --buffer-size 1024 --batch-size 256 --beam 5 --lenpen 1.0 \
    --remove-bpe=sentencepiece --no-progress-bar \
    > $FOUT
cat $FOUT | grep -P "^H" | cut -f 3- > $FOUT.out
cat $FOUT.out | python $SACRE_BLEU/sacrebleu.py $FTGT 
```

### <a name="usage_mtl_multitask"> MultiNMT with MultiTask Training </a>
Training (with BT, dynamic temperature, word-level MLM and span-level DAE)
```
python $FS_DIR/train.py $DATA_DIR --save-dir $OUTPUT_DIR \
    --arch transformer_mlm_vaswani_wmt_en_de_big \
    --task translation_mtl_multitask_curr \
    --lang-pairs "fr-en,de-en,fi-en" \
    --language-sample-temperature 5.0 \
    --min-language-sample-temperature 1.0 \
    --language-sample-warmup-epochs 5 \
    --language-temperature-scheduler "linear" \
    --data-bt $BT_DATA_DIR \
    --downsample-bt \
    --data-mono $MONO_DATA_DIR \
    --downsample-mono \
    --multitask-mlm \
    --lang-mlm fr,de,fi \
    --mlm-word-mask \
    --mlm-masking-ratio 0.15 \
    --multitask-dae \
    --lang-dae en \
    --dae-span-masking-ratio 0.35 \
    --dae-span-lambda 3.5 \
    --dae-max-shuffle-distance 3.0 \
    --dataset-impl mmap --share-all-embeddings --fp16 --ddp-backend=no_c10d \
    --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0 \
    --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --warmup-init-lr 1e-07 --min-lr 1e-09  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --max-update 100000 --update-freq 16 --log-interval 100 \
    --save-interval-updates 4000 --no-progress-bar
```
This is a X-EN system. To change it to EN-X or X-X, simply change the values for flags ```lang-pairs```, ```lang-mlm``` and ```lang-dae```. 
Training data is the same for all three system (i.e. EN-X, X-EN, X-X), constructed and organized in the way introduced in the next section.

For EN-X, change to: 
```
--lang-pairs "en-fr,en-de,en-fi" 
--lang-mlm "en" 
--lang-dae "fr,de,fi"
```

For X-X, change to: 
```
--lang-pairs "fr-en,de-en,fi-en,en-fr,en-de,en-fi" 
--lang-mlm "en,fr,de,fi" 
--lang-dae "en,fr,de,fi"
```

# <a name="data"> Data </a>

### Bitext Data
Bitext training data for all language pairs should be kept in ```$DATA_DIR```.

Train a shared SPM model with full (or sampled) corpus of all language pairs:

```
spm_train --input=$FSRC,$FTGT --model_prefix=spm_64k --vocab_size=64000 --character_coverage=1.0 
```

Encode each file:
```
spm_encode --model=spm_64k.model --output_format=piece < train.$SRC > train_spm.$SRC
```

Convert SPM vocab to fairseq vocab:
```
cut -f 1 spm_64k.vocab | tail -n +4 | sed "s/$/ 100/g" > $BIN_DIR/dict.src.txt
cp $BIN_DIR/dict.src.txt $BIN_DIR/dict.tgt.txt
```

Build binary data:
```
python $FS_DIR/preprocess.py \
    --task "translation" \
    --source-lang $SRC \
    --target-lang $TGT \
    --trainpref $TEXT_DIR/train_spm \
    --validpref $TEXT_DIR/dev_spm \
    --destdir $BIN_DIR \
    --dataset-impl 'mmap' \
    --padding-factor 1 \
    --workers 32 \
    --srcdict $BIN_DIR/dict.src.txt \
    --tgtdict $BIN_DIR/dict.tgt.txt
```

```dict.src.txt``` and ```dict.tgt.txt``` are required for multilingual training, and ```dict.$SRC.txt``` for each languages will be needed during testing.

E.g., for "fr-en,de-en" translation, folder ```$DATA_DIR``` should contain:

```
dict.src.txt
dict.tgt.txt
dict.fr.txt
dict.de.txt
train.de-en.de.bin
train.de-en.de.idx
train.de-en.en.bin
train.de-en.en.idx
train.fr-en.en.bin
train.fr-en.en.idx
train.fr-en.fr.bin
train.fr-en.fr.idx
```
The data can be directly used for training En-X and X-X system, by changing the --lang-pairs value to "en-fr,en-de" and "fr-en,de-en,en-fr,en-de" respectively.

### Back Translation

Back translations are sampled with bilingual systems.

```
python $FS_DIR/scripts/spm_encode.py --model $SPM_MODEL --inputs $SRC_FILE --outputs $FSRC
cat $FSRC | python $FS_DIR/interactive.py $TRAIN_DATA_DIR \
    --path $WORKING_DIR/$CKPT --source-lang $SRC --target-lang $TGT --task translation \
    --buffer-size 2048 --max-tokens 60000 --batch-size 2048 \
    --sampling --beam 1 --nbest 1 --lenpen 1.0 --remove-bpe=sentencepiece \
> $FTGT
cat $FTGT | grep -P "^H" | cut -f 3- > $FTGT.trans
```

BT data will be spm encoded and binarized the same way as the bitext data. 
BT data for all language pairs should be kept in a same folder, which is passed to ```--data-bt```.

Note that during training, bitext data and BT data will be mixed and shuffled to build each training batch. 
So it can also be done by combining bitext and BT data during preprocessing. 
The ```--data-bt``` flag is for convenience only. 

### Monolingual Data

```
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
```

E.g., for "fr-en,de-en" translation, folder ```--data-mono``` should contain:

```
train.de.bin
train.de.idx
train.fr.bin
train.fr.idx
train.en.bin
train.en.idx
```

# <a name="tasks"> Tasks </a>

### <a name="tasks_mtl"> translation_mtl </a>
Task for multilingual translation, supports:
* multilingual training (should do offline during preprocessing)
* back translation

Supports all arguments in Named Arguments - [Multilingual Training](#args_mtl), e.g. multilingual with back translation:
```
--task translation_mtl --lang-pairs "fr-en,de-en" --data-bt $BT_DATA_DIR --downsample-bt 
```

### <a name="tasks_mtl_ols"> translation_mtl_ols </a>
task for multilingual translation with online data sampling, supports:
* multilingual training 
* online sampling
* online sampling with dynamic temperature
* back translation

Supports all arguments in Named Arguments - [Multilingual Training](#args_mtl) and [Data Scheduling](#args_data_curr), e.g. online sampling with static T=5:

```
--task translation_mtl_ols \
--lang-pairs "fr-en,de-en" \
--language-sample-temperature 5.0 \
--language-temperature-scheduler "static" 
```

### <a name="tasks_mtl_multitask"> translation_mtl_multitask_curr </a>
task for multilingual translation with multitask and curriculum learning, supports:

* multilingual training 
* online sampling
* multitask objective: masked language model
* multitask objective: denoising auto-encoder
* data scheduling: dynamic sampling temperature 
* task scheduling: dynamic multitask weights and noising ratio

Note: to use this task, model must be ```transformer_mlm```, e.g. ```--arch transformer_mlm_vaswani_wmt_en_de_big```.

Example usage refer to [Usage - MultiNMT with MultiTask Training](#usage_mtl_multitask).

# <a name="named_args"> Named Arguments </a>

### <a name="args_mtl"> Multilingual Training </a>
|                   |                |
| ----------------- |:---------------| 
| --lang-pairs      | list of multilingual language pairs (comma-separated) <br><br> e.g.: "en-de,en-fr,de-fr" |
| --encoder-langtok | add EOS in source sentence with source or target language token <br><br> default: "tgt", choices=['src', 'tgt']|
| --decoder-langtok | replace BOS in target sentence with target language token <br><br> default: False, action='store_true' |
| --data-bt         | path to back translation data directory |
| --downsample-bt   | downsample bt to match the amount of bitext data in each epoch <br><br> default: False, action='store_true' |

### Multitasking
|                     |                |
| -----------------   |:---------------| 
| --data-mono         | path to mono data directory |
| --downsample-mono   | downsample mono to match the amount of parallel data in each epoch |
| --bpe-cont-marker   | used to find word boundary for word-level noising <br><br> default: sentencepiece, choices=[sentencepiece, bpe, token] |
| --static-noising    | use same noising for same example in each epoch <br><br> default: False |

### Masked Language Model
|                     |                |
| -----------------   |:---------------| 
| --multitask-mlm     | use MaskedLM objective together with MT cross-entropy |
| --lang-mlm          | comma-separated list of monolingual languages for MLM <br><br> e.g.: "en,de,fr" |
| --mlm-masking-ratio | masking ratio for MaskedLM <br><br> default: 0.15 |
| --mlm-masking-prob  | probability of replacing the masked token with special token \<mask\> <br><br> default: 0.8 |
| --mlm-random-token-prob | probability of replacing the masked token with a random token <br><br> default: 0.1 |
| --mlm-word-mask     | use word-level random masking for MaskedLM <br><br> default: False (i.e. token-level) |
| --mlm-span-mask     | use span masking for MaskedLM <br><br> default: False (i.e. token-level) |
| --mlm-span-lambda   | lambda of poisson distribution for span length sampling <br><br> default: 3.5 |
| --mlm-activation-fn | activation function to use in MLM | 
| --share-encoder     | share encoder for seq2seq and mlm task <br><br> default: True |

### Denoising Auto-Encoder
|                            |                |
| -----------------          |:---------------| 
| --multitask-dae            | use DAE objective together with MT cross-entropy |
| --lang-dae                 | comma-separated list of monolingual languages for DAE <br><br> e.g.: "en,de,fr" |
| --dae-max-shuffle-distance | maximum shuffle distance for DAE <br><br> default: 3.0 |
| --dae-dropout-prob         | word dropout probability for DAE  (word dropout) <br><br> default: 0.0 |
| --dae-blanking-prob        | word blanking probability for DAE (word blank) <br><br> default: 0.0 |
| --dae-blanking-with-mask   | word blanking with \<mask\> token instead of \<unk\> <br><br> default: False |
| --dae-span-masking-ratio   | span masking ratio for DAE <br><br> default: 0.35 |
| --dae-span-lambda          | lambda of poisson distribution for span length sampling <br><br> default: 3.5 |

### <a name="args_data_curr"> Data Scheduling </a> 
|                                   |                |
| -----------------                 |:---------------| 
| --language-sample-temperature     | sampling temperature for multi-languages <br><br> default: 1.0 |
| --language-upsample-max           | upsample to make the max-capacity language a full set <br><br> default: False (i.e. up and downsample to keep a fixed total corpus size) |
| --language-temperature-scheduler  | sampling temperature scheduler <br><br> default: static, choices=[static, linear] |
| --min-language-sample-temperature | minimum (starting) sampling temperature <br><br> default: 1.0 |
| --language-sample-warmup-epochs   | warmup epochs for language sampling scheduler <br><br> default: 0 |

For dynamic (linear) temperature:

T(k) = (T - T_0) / N + T_0

* k:    current epoch
* T:    --language-sample-temperature
* T_0:  --min-language-sample-temperature
* N:    --language-sample-warmup-epochs

### Multitask Weight Scheduling
|                       |                |
| -----------------     |:---------------| 
| --multitask-scheduler | multitask weight scheduler <br><br> default=static, choices=[static, linear] |
| --mlm-alpha           | weight for mlm objective <br><br> default: 1.0 |
| --mlm-alpha-min       | minimum weight for mlm objective <br><br> default: 1.0 |
| --mlm-alpha-warmup    | warmup epochs for mlm objective <br><br> default: 1 |
| --dae-alpha           | weight for dae objective <br><br> default: 1.0 |
| --dae-alpha-min       | minimum weight for dae objective <br><br> default: 1.0 | 
| --dae-alpha-warmup    | warmup epochs for dae objective <br><br> default: 1 |

W(k) = W - (W - W_0) / N
* k:   current epoch
* W:   alpha
* W_0: alpha-min
* N:   alpha-warmup

### Noising Ratio Scheduling
|                                    |                |
| -----------------                  |:---------------| 
| --mlm-masking-ratio-min            | minimal (starting) masking ratio for MaskedLM <br><br> default: 0.15 |
| --mlm-masking-ratio-warmup-epochs  | warmup epochs for masking ratio scheduler <br><br> default: 1 |
| --mlm-masking-ratio-scheduler      | MaskedLM masking ratio scheduler <br><br> default: static, choices=[static, linear] |
| --dae-span-masking-ratio-min       | minimal (starting) span masking ratio for DAE <br><br> default: 0.35 |
| --dae-span-masking-ratio-warmup-epochs | warmup epochs for span masking ratio scheduler <br><br> default: 1 |
| --dae-span-masking-ratio-scheduler | DAE span masking ratio scheduler <br><br> default: static, choices=[static, linear] |

# <a name="scripts"> Scripts </a>
|                           |                |
| -----------------         |:---------------| 
| train.sh                  | training script |
| test.sh                   | testing script |
| train_xxen_multitask.yaml | config file for X-EN system with multitask training, dynamic temperature sampling and BT |
| test_xxen_multitask.yaml  | config file for testing on all pairs of the above system |
| data/data_gen.sh          | build binarized bitext data |
| data/data_gen_mono.sh     | build binarized monolingual and BT data |
| data/filter_bitext.py     | rule-based filtering for parallel data |
| data/filter_mono.py       | rule based filtering for monolingual data |
| data/sample_offline_up.py | temperature based offline upsampling |

# <a name="codes"> Codes </a>
Description of files with new features or major modifications.

### tasks
|                        |                |
| -----------------      |:---------------| 
| translation_mtl.py     | task for multilingual translation <br> see [translation_mtl](#tasks_mtl) |
| translation_mtl_ols.py | task for multilingual translation with online sample <br> see [translation_mtl_ols](#tasks_mtl_ols) |
| translation_mtl_multitask_curr.py | task for multilingual translation with multitask learning <br> see [translation_mtl_multitask_curr](#tasks_mtl_multitask) |

### models
|                        |                |
| -----------------      |:---------------| 
| transformer_mlm.py     | transformer with language model head on encoder side |

### criterions
|                        |                |
| -----------------      |:---------------| 
| mlm_loss.py            | loss for masked language model <br> only compute loss for masked positions |

### data
|                            |                |
| -----------------          |:---------------| 
| langpair_dataset_loader.py | data loader for multilingual bitext, bt and monolingual data |
| language_pair_langid_dataset.py | dataset wrapper for language pairs with language id appended |
| masking.py                 | masking schemes for MLM and DAE <br> * RandomTokenMasking: token-level masking for MLM; <br> * RandomWordMasking: word-level masking for MLM; <br> * SpanTokenMasking: span masking for MLM; <br> * TextInfilling: span mask for DAE (BART-style); <br>
| masking_dataset.py         | dataset wrapper for MLM |
| noising.py                 | add support for sentencepiece | 
| noising_dataset.py         | dataset wrapper for DAE |
| dictionary.py              | add special_symbol_index, needed for determining word boundary with sentencepiece |
| resampling_dataset.py      | do randomly sampling at each epoch |
| concat_dataset.py          | add epoch update and support updating sample ratios |