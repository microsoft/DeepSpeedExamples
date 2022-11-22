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
### use 8GPU to train Imagenet 
### use "deepspeed --help" to know more about deepspeed
mkdir out_imagenet
deepspeed  --include worker-0:0,1,2,3,4,5,6,7 --master_port 60000  main_imagenet.py /blob/data/ImageNetData/ --seed 2345    \
                --deepspeed_config config/ds_config_imagenet.json  \
                --deepspeed    --random_ltd          \
                -a vits16r224  \
                --img_size 224  --print-freq 100 --batch-size 64 \
                --lr 5e-05 --scheduler constant --dummy \
                --epochs 14 \
                --out_dir out_imagenet | tee -a out_imagenet/training.log