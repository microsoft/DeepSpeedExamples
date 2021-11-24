OOO=increase3em5_MAYDEL
MASTER_PORT=12401
GPU=1

mkdir $OOO

for TSK in cola # sst2 mrpc wnli rte qnli qqp mnli stsb
do
echo $((MASTER_PORT++))
echo $((GPU++))

if [ $TSK == wnli ] || [ $TSK == mrpc ]
then
    EPOCH_NUM=5
else
    EPOCH_NUM=2
fi

if [ $TSK == qqp ] || [ $TSK == mnli ]
then
    TEST_JSON=test_long.json
else
    TEST_JSON=test.json
fi

PORT=$((MASTER_PORT)) # +GPU))

#  --do_predict \
#  --do_eval \
#  --per_device_eval_batch_size 1 \
#  --do_train \
#  --per_device_train_batch_size 32 \

learn=3e-5
trnbsz=24
evalbsz=1
echo $learn
echo $trnbsz
echo $evalbsz

CUDA_VISIBLE_DEVICES=$GPU nohup python -m torch.distributed.launch \
  --master_port $PORT \
  --nproc_per_node 1 run_glue.py \
  --task_name ${TSK} \
  --model_name_or_path bert-base-uncased \
  --max_seq_length 128 \
  --do_train \
  --per_device_train_batch_size $trnbsz \
  --do_eval \
  --per_device_eval_batch_size $evalbsz \
  --learning_rate 3e-5 \
  --num_train_epochs $EPOCH_NUM \
  --output_dir ./$OOO/$TSK/ \
  --fp16 \
  --warmup_steps 2 \
  --deepspeed test.json > ./${OOO}/${TSK}_trnbsz${trnbsz}_evalbsz${evalbsz}_t11.out 2> ./${OOO}/${TSK}_trnbsz${trnbsz}_evalbsz${evalbsz}_t11.err &

done
