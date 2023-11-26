#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=1 demo.py --cfg ./config/demo/demo_224_real.yaml
# ./config/demo/demo_224_amazon.yaml    
# ./config/demo/demo_224_real.yaml

CUDA_VISIBLE_DEVICES=15  python -m torch.distributed.launch --master_port 8888 \
--nproc_per_node=1  demo.py --cfg ./config/demo/demo_224_real.yaml --permute \
--cpt /vc_data/users/xwu/Model3d/LEAP/output1/8v5asrdb4wzn55atzrdy5csblasg1jdu.tar
#--cpt /vc_data/users/xwu/Model3d/LEAP/output1/sfvznslazrwrrof8fv7uy23myc0oizhx.tar