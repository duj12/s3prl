#/usr/bin/bash
stage=1

if [ $stage -eq 0 ]; then
python3 -u preprocess/generate_len_for_bucket.py -i /data/megastore/Datasets/ASR/English/LibriTTS/ -a wav
echo 0 1 2 3 4

fi

if [ $stage -eq 1 ]; then
CUDA_VISIBLE_DEVICES=6  \
	python -u run_pretrain.py -u distiller -g pretrain/distiller/config_model.yaml -n distill-hubert -e result/pretrain/distill-hubert/states-40000.ckpt

fi

