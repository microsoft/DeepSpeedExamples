# python main_imagenet.py /blob/data/ImageNetData/ --seed 2345    \
#                 --deepspeed_config config/ds_config.json  \
#                 --deepspeed              \
#                 --dist-url 'tcp://127.0.0.1:65402'  --dist-backend 'nccl' \
#                 --multiprocessing-distributed            \
#                 -a vits16r224 --world-size 1 --rank 0 \
#                 --img_size 224  --print-freq 100 --batch-size 256 \
#                 --lr 5e-05 --scheduler constant --dummy   \
#                 --epochs 14 \
#                 --out_dir onstant_tokenInterval0.8_tokenInit66_pass1-21layer | tee -a train_log.txt

mkdir out_imagenet
deepspeed  main_imagenet.py /blob/data/ImageNetData/ --seed 2345    \
                --deepspeed_config config/ds_config.json  \
                --deepspeed              \
                -a vits16r224  \
                --img_size 224  --print-freq 100 --batch-size 256 \
                --lr 5e-05 --scheduler constant --dummy  --random_ltd \
                --epochs 14 \
                --out_dir out_imagenet | tee -a out_imagenet/training.log