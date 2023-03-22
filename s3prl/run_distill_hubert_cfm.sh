#/usr/bin/bash
stage=$1

if [ $stage -eq 0 ]; then
python3 -u preprocess/generate_len_for_bucket.py -i /data/megastore/Datasets/ASR/English/LibriTTS/ -a wav
echo 0 1 2 3 4

fi

if [ $stage -eq 1 ]; then
CUDA_VISIBLE_DEVICES=0  \
	python -u run_pretrain.py -u distiller -g pretrain/distiller/config_model_cfm.yaml -n distill-hubert-cfm

fi


if [ $stage -eq 2 ]; then
CUDA_VISIBLE_DEVICES=1  \
	python -u run_pretrain.py -u distiller -c pretrain/distiller/config_runner_s.yaml \
	        -g pretrain/distiller/config_model_s.yaml -n distill-hubert-s

fi

if [ $stage -eq 3 ]; then
CUDA_VISIBLE_DEVICES=2  \
	python -u run_pretrain.py -u distiller -c pretrain/distiller/config_runner_s.yaml \
	        -g pretrain/distiller/config_model_s1.yaml -n distill-hubert-s1

fi

if [ $stage -eq 4 ]; then
CUDA_VISIBLE_DEVICES=3  \
	python -u run_pretrain.py -u distiller -c pretrain/distiller/config_runner_m.yaml \
	        -g pretrain/distiller/config_model_m.yaml -n distill-hubert-m

fi

if [ $stage -eq 5 ]; then
CUDA_VISIBLE_DEVICES=6  \
	python -u run_pretrain.py -u distiller -c pretrain/distiller/config_runner_s.yaml \
	        -g pretrain/distiller/config_model_s2.yaml -n distill-hubert-s2

fi

if [ $stage -eq 6 ]; then
CUDA_VISIBLE_DEVICES=7  \
	python -u run_pretrain.py -u distiller -c pretrain/distiller/config_runner_s.yaml \
	        -g pretrain/distiller/config_model_s3.yaml -n distill-hubert-s3
fi