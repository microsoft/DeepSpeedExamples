CUDA_VISIBLE_DEVICES=8,9,10,11,12, python -m torch.distributed.launch --master_port 8888 --nproc_per_node=1 \
eval.py --cfg ./config/omniobject3d/eval_224.yaml