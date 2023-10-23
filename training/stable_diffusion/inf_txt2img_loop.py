from local_pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import StableDiffusionPipeline as StableDiffusionPipelineBaseline
import torch
import os

seed = 123450011
new_model = "new_sd-distill-v21-10k-1e"
baseline_model = "stabilityai/stable-diffusion-2-1-base"
image_out_dir = "out/"

if not os.path.exists(image_out_dir):
        os.makedirs(image_out_dir)
        print(f"Directory '{image_out_dir}' has been created to store the generated images.")
else:
        print(f"Directory '{image_out_dir}' already exists and stores the generated images.")


prompts = ["A boy is watching TV",
           "A photo of a person dancing in the rain",
           "A photo of a boy jumping over a fence",
           "A photo of a boy is kicking a ball",
           "A beach with a lot of waves on it",
           "A road that is going down a hill",
           "3d rendering of 5 tennis balls on top of a cake",
           "A person holding a drink of soda",           
           "A person is squeezing a lemon",
           "A person holding a cat"]


for prompt in prompts:
    #--- new image
    pipe_new = StableDiffusionPipeline.from_pretrained(new_model, torch_dtype=torch.float16).to("cuda")
    generator = torch.Generator("cuda").manual_seed(seed)
    image_new = pipe_new(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]

    image_new.save(image_out_dir+"NEW__seed_"+str(seed)+"_"+prompt[0:100]+".png")

    #--- baseline image
    pipe_baseline = StableDiffusionPipelineBaseline.from_pretrained(baseline_model, torch_dtype=torch.float16).to("cuda")
    generator = torch.Generator("cuda").manual_seed(seed)                                              
    image_baseline = pipe_baseline(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]                                                                                 


    image_baseline.save(image_out_dir+"BASELINE_seed_"+str(seed)+"_"+prompt[0:100]+".png")
