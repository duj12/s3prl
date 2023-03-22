#/usr/bin/bash
stage=$1

if [ $stage -eq -1 ]; then
echo "1 2 3 4 5 6 7" | python3 -u preprocess/generate_len_for_bucket.py \
  -i /data/megastore/Datasets/ASR/English/LibriSpeech/data/LibriSpeech -a flac

fi

if [ $stage -eq 0 ]; then
for dataset in ASR_36k; do
for sub in dev test train; do
 python3 -u preprocess/generate_len_for_bucket.py \
  -s data/$dataset/$sub/wav.scp   -t data/$dataset/$sub/text \
  -o data/$dataset -n $sub
done
done
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


# distiller模型
if [ $stage -eq 4 ]; then
CUDA_VISIBLE_DEVICES=1  \
	python -u run_downstream.py -m train -n distiller-s1last-asr -u distiller_local -d asr -a --upstream_no_pred false \
	-k /data/megastore/Projects/DuJing/code/s3prl/s3prl/data/pretrained_models/distiller-s1/states-450000.ckpt
  --upstream_feature_selection last_hidden_state
fi


# distiller模型
if [ $stage -eq 5 ]; then
CUDA_VISIBLE_DEVICES=7  \
	python -u run_downstream.py -m train -c downstream/asr/config_aishell.yaml \
	  -n aishell-distiller-s-asr -u distiller_local -d asr -a --upstream_no_pred true \
  	-k data/pretrained_models/distiller-s/states-345000.ckpt

fi

# distiller模型
if [ $stage -eq 6 ]; then
CUDA_VISIBLE_DEVICES=6  \
	python -u run_downstream.py -m train -c downstream/asr/config_aishell.yaml \
  -n aishell-hubert-cfm-asr -u hubert_local  -d asr -a \
	-k data/pretrained_models/ASR_80fps/hubert_ft_v1.pt

fi