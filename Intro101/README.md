
good practices
  * experiment directory saving training metadata
  * pytests

data
  * what is masked LM
  * what does the code do (link to code)

model
  * core params for transformer model (e.g., #layers, attn)

how to run
  * launching on CPU (slow) launch on single GPU (fast)
  * different train params

------

deepspeed additions
  * deepspeed.init, training loop, ckpt changes

launching across multiple GPUs

fp16
  * how to enable
  * show memory reduction when enabled via nvidia-smi
  * brief overview of how fp16 training works (e.g., loss scaling)

zero
  * introduce how zero reduces memory
  * introduce zero offload
  * update config to use z1 + offload to showcase a model that can only run with offload enabled
