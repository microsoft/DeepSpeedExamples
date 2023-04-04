import deepspeed
import torch
import os
from deepspeed.accelerator import get_accelerator
import time

from diffusers import DiffusionPipeline

prompt = "a dog on a rocket"

model = "prompthero/midjourney-v4-diffusion"
local_rank = int(os.getenv("LOCAL_RANK", "0"))
device = torch.device(f"cuda:{local_rank}")
world_size = int(os.getenv('WORLD_SIZE', '1'))
generator = torch.Generator(device=torch.cuda.current_device())

pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half)
pipe = pipe.to(device)

generator.manual_seed(0xABEDABE7)
baseline_image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
baseline_image.save(f"baseline.png")

# NOTE: DeepSpeed inference supports local CUDA graphs for replaced SD modules.
#       Local CUDA graphs for replaced SD modules will only be enabled when `mp_size==1`
pipe = deepspeed.init_inference(
    pipe,
    mp_size=world_size,
    dtype=torch.half,
    replace_with_kernel_inject=True,
    enable_cuda_graph=True if world_size==1 else False,
)

times_list = []

prompt_list = [
    "a dog on a rocket",
    "a dog on a table",
    "a dog in a field",
    "a grocery store full of produce",
    "a city with bright lights",
    ]

generator.manual_seed(0xABEDABE7)

for prompt in prompt_list:
    get_accelerator().synchronize()
    start = time.time()

    deepspeed_image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]

    get_accelerator().synchronize()
    end = time.time()
    times_list.append((end - start))

    prompt_file_name = prompt.replace(" ", "_")
    deepspeed_image.save(f"deepspeed_{prompt_file_name}.png")

print(times_list)
