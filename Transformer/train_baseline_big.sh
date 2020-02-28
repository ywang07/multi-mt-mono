export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nvidia-smi
python -c "import torch; print('pytorch version=' + torch.__version__);  print('MT-pytorch version= v0.1.0 ' )"
python train.py $1 --save-dir $2 --skip-invalid-size-inputs-valid-test  --arch transformer_vaswani_wmt_en_de_big  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --share-all-embeddings --dropout 0.3 --max-tokens 5000  --fp16 --max-update 20000 --log-interval 100 --ddp-backend=no_c10d  --update-freq 16

