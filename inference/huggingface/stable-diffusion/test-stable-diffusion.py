import deepspeed
import torch
import os

from diffusers import DiffusionPipeline

prompt = "a dog on a rocket"

model = "prompthero/midjourney-v4-diffusion"
local_rank = int(os.getenv("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{local_rank}")
world_size = int(os.getenv('WORLD_SIZE', '4'))

pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half)
pipe = pipe.to(device)

baseline_image = pipe(prompt, guidance_scale=7.5).images[0]
baseline_image.save(f"baseline.png")

pipe = deepspeed.init_inference(
    pipe,
    mp_size=1,
    dtype=torch.half,
    replace_method="auto",
    replace_with_kernel_inject=False,
    enable_cuda_graph=False,
)

deepspeed_image = pipe(prompt, guidance_scale=7.5).images[0]
deepspeed_image.save(f"deepspeed.png")
