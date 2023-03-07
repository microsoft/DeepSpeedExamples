set -x

for gpus in `echo "1 2 4"`; do
    for batch in `echo "1 2 4 8 16 32 64 128 256 320 512 832"`; do
        deepspeed --num_gpus $gpus DeepSpeedExamples/inference/huggingface/text2text-generation/test-text2text.py --model "t5-11b" --ds_inference --batch_size $batch --test_performance
    done
done