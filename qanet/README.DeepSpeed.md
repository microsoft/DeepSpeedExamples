This is an example folder that uses QANet to demonstrate how to port models to DeepSpeed. The approach that we took in this example is to create a DeepSpeed version of affected python code files, rather than editing the original files with if/then/else patterns to guard DeepSpeed codes. Consequently, you can do a diff of the affected files to conveniently see the required changes. Specifically, for porting to DeepSpeed we found that only two files needed to be replicated:

  1) main entry file (main.py) -> main_deepspeed.py

  2) model file (model_train/model_train.py) -> model_train/model_train_deepspeed.py

Additionally, we provide four shell scripts to illustrate different experiments:

1. run_baseline.sh: Run the baseline qanet with batch size of 32 on one GPU. Add --multi_gpu on command line if GPU RAM is too small to use spread the batch size across multiple GPUs on the node.

2. run_deepspeed_bsz32_warmup.sh: Run DeepSpeed qanet with single node, multi-gpu, batch size 32, and Warmup LR schedule.

3. run_deepspeed_bsz512_warmup.sh: Run DeepSpeed qanet with multi-node, multi-gpu, batch size 512, and warmup LR schedule.

4. run_deepspeed_bsz512_1cycle.sh: Run DeepSpeed qanet with multi-node, multi-gpu, batch size 512, and 1cycle LR schedule.
