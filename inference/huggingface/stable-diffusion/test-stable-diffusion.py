import deepspeed
import torch
import os
from local_pipeline_stable_diffusion import StableDiffusionPipeline

prompt = "a dog on a rocket"

model = "prompthero/midjourney-v4-diffusion"
local_rank = int(os.getenv("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{local_rank}")
world_size = int(os.getenv('WORLD_SIZE', '1'))
generator = torch.Generator(device=torch.cuda.current_device())

pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.half)
pipe = pipe.to(device)

seed = 0xABEDABE7
generator.manual_seed(seed)
baseline_image = pipe(prompt, guidance_scale=7.5, generator=generator, optimized_iterations=0).images[0]
baseline_image.save(f"baseline.png")


# NOTE: DeepSpeed inference supports local CUDA graphs for replaced SD modules.
#       Local CUDA graphs for replaced SD modules will only be enabled when `mp_size==1`
pipe = deepspeed.init_inference(
    pipe,
    mp_size=world_size,
    dtype=torch.half,
    replace_with_kernel_inject=True,
    enable_cuda_graph=False if world_size==1 else False,
    )

generator.manual_seed(seed)
deepspeed_image = pipe(prompt, guidance_scale=7.5, generator=generator, optimized_iterations=0.20).images[0]
deepspeed_image.save(f"deepspeed.png")
