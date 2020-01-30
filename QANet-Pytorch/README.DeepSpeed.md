This is an example folder that uses QANet to demonstrate how to port models to DeepSpeed. The approach that we took in this example is to create a DeepSpeed version of affected python code files, rather than editing the original files with if/then/else patterns to guard DeepSpeed codes. Consequently, you can do a diff of the affected files to conveniently see the required changes. Specifically, for porting to DeepSpeed we found that only two files needed to be replicated:

  1) main entry file (main.py)

  2) model file (model_train/model_train.py)

We have illustrated three scenarios of DeepSpeed porting, which results in three new entry files

  1) main_ds.py: Using the following DeepSpeed features
    1.1 DDP distributed model training
    1.2 Optimizer
    1.3 LR scheduler
    1.4 Distributed data loader

  2) main_ds_adam.py: Scenario 1 without DeepSpeed LR scheduler

  3) main_ds_warmup.py: Scenario 1 without DeepSpeed optimizer 

  Corresponding test bash scripts and DeepSpeed json configuration files are also provided:

  1) main_ds.py: 
             run_ds_1gpu.sh &  
             run_ds_16gpu.sh, 

  2) main_ds_adam.py: run_ds_adam.sh and short_deepspeed_adam.json

  3) main_ds_warmup.py: run_ds_warmup.sh and short_deepspeed_warmup.json
