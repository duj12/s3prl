#/usr/bin/bash
stage=$1

if [ $stage -eq 0 ]; then
echo "1 2 3 4 5 6 7" | python3 -u preprocess/generate_len_for_bucket.py \
  -i /data/megastore/Datasets/ASR/English/LibriSpeech/data/LibriSpeech -a flac

fi
#hubert-cfm模型
if [ $stage -eq 1 ]; then
CUDA_VISIBLE_DEVICES=0  \
	python -u run_downstream.py -m train -n hubert-cfm-asr -u hubert_local  -d asr -a \
	-k /data/megastore/Projects/DuJing/code/s3prl/s3prl/data/pretrained_models/ASR_80fps/hubert_ft_v1.pt

fi

# distiller模型
if [ $stage -eq 2 ]; then
CUDA_VISIBLE_DEVICES=2  \
	python -u run_downstream.py -m train -n distiller-s-asr -u distiller_local -d asr -a --upstream_no_pred true \
	-k /data/megastore/Projects/DuJing/code/s3prl/s3prl/data/pretrained_models/distiller-s/states-345000.ckpt

fi

# distiller模型
if [ $stage -eq 3 ]; then
CUDA_VISIBLE_DEVICES=3  \
	python -u run_downstream.py -m train -n distiller-s1-asr -u distiller_local -d asr -a --upstream_no_pred true \
	-k /data/megastore/Projects/DuJing/code/s3prl/s3prl/data/pretrained_models/distiller-s1/states-450000.ckpt

fi

