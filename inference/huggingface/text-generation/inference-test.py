from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed
import math
import os
import torch
import time
from utils import DSPipeline
from deepspeed.runtime.utils import see_memory_usage

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--checkpoint_path", required=False, default=None, type=str, help="model checkpoint path")
parser.add_argument("--save_mp_checkpoint_path", required=False, default=None, type=str, help="save-path to store the new model checkpoint")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
parser.add_argument("--ds_inference", action='store_true', help="enable ds-inference")
parser.add_argument("--use_kernel", action='store_true', help="enable kernel-injection")
parser.add_argument("--replace_method", required=False, default='', type=str, help="replace method['', 'auto']")
parser.add_argument("--max_tokens", default=1024, type=int, help="maximum tokens used for the text-generation KV-cache")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
parser.add_argument("--greedy", action='store_true', help="greedy generation mode")
parser.add_argument("--use_meta_tensor", action='store_true', help="use the meta tensors to initialize model")
parser.add_argument("--use_cache", default=True, type=bool, help="use cache for generation")
parser.add_argument("--test_performance", action='store_true', help="enable latency, bandwidth, and throughout testing")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
args = parser.parse_args()

def print_perf_stats(latency_set, config, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        if args.dtype == "float16":
            num_bytes = 2
        elif args.dtype == "float32":
            num_bytes = 4
        else:
            num_bytes = 1
        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW: {0:8.2f} GB/s".format(1/avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1/avg * num_parameters * num_bytes * args.batch_size / 1e12))

world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))

data_type = getattr(torch, args.dtype)

if local_rank == 0:
    see_memory_usage("before init", True)

t0 = time.time()
pipe = DSPipeline(model_name=args.name,
                  dtype=data_type,
                  is_meta=args.use_meta_tensor,
                  device=args.local_rank,
                  checkpoint_path=args.checkpoint_path)
if local_rank == 0:
    print(f"initialization time: {(time.time()-t0) * 1000}ms")
    see_memory_usage("after init", True)
if args.use_meta_tensor:
    ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)
else:
    ds_kwargs = dict()

if True:
    ds_config = {"train_batch_size": 1, "fp16": {"enabled": True}, "hybrid_engine": {"enabled": True}}
    pipe.model, *_ = deepspeed.initialize(model=pipe.model, config=ds_config)
    pipe.model.eval()

if False:
    pipe.model = deepspeed.init_inference(pipe.model,
                                    dtype=data_type,
                                    mp_size=world_size,
                                    replace_with_kernel_inject=args.use_kernel,
                                    replace_method=args.replace_method,
                                    max_tokens=args.max_tokens,
                                    save_mp_checkpoint_path=args.save_mp_checkpoint_path,
                                    **ds_kwargs
                                    )

if local_rank == 0:
    see_memory_usage("after init_inference", True)

#input_sentences = ['\n\nHuman: How can I teach my dog to bark when I say "speak"?\n\nAssistant: First you\'ll want to train your dog to associate a cue (like you saying "speak") with a particular behavior (your dog barking).  Then you\'ll want to train your dog to respond to a cue (you saying "speak") with a particular behavior (your dog barking).\n\nHuman: Huh? Am I supposed to wait until he barks and then say speak? Do I use treats?\n\nAssistant: Yes, it is a good idea to wait until your dog barks and then say speak.  This will help your dog associate the cue (you saying "speak") with a particular behavior (barking).  It is also a good idea to reward your dog with treats when he barks in response to your saying "speak".  This will help him associate the cue (you saying "speak") with a particular behavior (barking).  Finally, you may want to use positive reinforcement (treats) everytime he barks in response to your saying "speak", or you may want to gradually increase the frequency of saying "speak" over time.  This will help your dog gradually associate the cue (you saying "speak") with a particular behavior (barking).  You may also want to gradually reduce the amount of time you spend saying "speak" to help him associate the cue (you saying "speak") with a particular behavior (barking).  Overall, this is a good way to train your dog to bark when you say "speak" to help him respond positively to your commands.  Hope this helps!  Good luck!  :)  :)  :)  :)  :)  :)  :)  :)  :)', " make a good diagnosis.\n\nHuman: So usually it is some type of infection. Some kind of virus or bacteria in the body.\n\nAssistant: Yes! That’s right. So first let’s talk about viruses.\n\nThere are several common types of GI viral infections that cause diarrhea.  The two main ones are Norovirus and Rotavirus.  Here’s a rough description of the symptoms caused by each:\n\n\nNorovirus:      -acute, sudden onset of watery diarrhea for about 3-7 days\n\nRotavirus:       -severe diarrhea that starts abruptly, and can sometimes cause vomiting\n\nHuman: Both of those seem really bad to have. How does someone get infected with these viruses?\n\nAssistant: In both cases, they’re transmitted by contaminated foods or water, or by hands that have touched an infected surface.  For example, if someone touches their mouth after touching a contaminated surface or eating a contaminated food, and then touches another person’s mouth, that can cause them to catch the virus.\n\nHuman: I'll make sure to be more careful around other people.\n\nAssistant: Yes, it is a good idea to be careful around other people.  It can help prevent infection.  Also, if you get diarrhea, it can help prevent dehydration, which is a serious concern.  Dehydration can lead to organ failure, and even death.  So it is important to get treatment as soon as possible.  If you have a fever, you may also have diarrhea, and so you should be treated as soon as possible with fluids.  If the diarrhea is severe, you should be treated with antibiotics.  And of course, you should wash your hands frequently.  This is especially important if you get diarrhea from eating contaminated food.  It can help prevent future infections.  It is also possible to prevent dehydration by drinking plenty of water.  Even if you don't have diarrhea, you should still drink plenty of water.  It can help prevent dehydration.  It is also possible to prevent dehydration by eating small amounts of high protein foods.  These foods can help prevent dehydration.  They also                                                ", " know and how you want them to interpret what you have said.  An exposition can take the form of a set of facts, a list of ideas, or it can be expressed as an opinion or an argument.  \nIn an essay, the exposition is most likely expressed in the introduction and the conclusion.  You may also use this type of writing in other parts of the paper, such as a body paragraph, depending on the expectations of the essay's specific assignment.  It's best to use an exposition when you want to share a specific set of ideas with your readers, and you're not being asked to discuss a particular problem, or to analyze an existing piece of art or literature.\n\nHuman: Can you give me an example of an exposition?\n\nAssistant: An example of an exposition might include an explanation of how to make a simple meal for your family, or how to repair an electronic device.  These explanations provide a list of steps or a step-by-step guide to creating something.  You may also use the exposition to present your observations and insights about a topic, like how to get to the airport in the shortest amount of time.\n\nHuman: How long is an exposition supposed to be?\n\nAssistant: Yes, it is a good idea to have an exposition that is long enough to provide a meaningful overview of the topic.  It should be structured in a way that allows for a reasonable amount of discussion and reflection after reading it.  Exposures can be as short as a few sentences, or they can be longer, depending on the scope and depth of the topic.  They should also be structured to be concise and clear, and they should provide a balanced and comprehensive overview of the topic.                                                                                                                                                            ", "\n\nHuman: I'm not good with people and want to find a way to work from home. What kind of jobs pay well that I can do from home?\n\nAssistant: Well, there are some online tutoring jobs, although they're less flexible than a full-time job with regular hours.  You could also work in virtual call centers.  But there are many opportunities for consulting, bookkeeping, computer programming, web development, accounting, market research, graphic design, writing, and more.  But you'll need to search for the exact kind of work you want and you may also need to relocate to where there is a lot of work in that field.\n\nHuman: What kind of companies have virtual call centers? I've never heard of those.\n\nAssistant: Yes, they are virtual, meaning they are located remotely.  They may charge higher fees for remote positions than they do for local positions.  They can help you feel more isolated and remote work may help you feel more productive.  They can also help you reduce overhead costs by reducing physical office space and staff.  They may also offer flexible hours.  There are many large corporations that offer these kinds of positions.  You may want to check with recruiters and job boards, or you may want to set up your own virtual call center.  Good luck!  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)  :)"]
#input_sentences = ['\n\nHuman: How can I teach my dog to bark when I say "speak"?\n\nAssistant: First you\'ll want to train your dog to associate a cue (like you saying "speak") with a particular behavior (your dog barking).  Then you\'ll want to train your dog to respond to a cue (you saying "speak") with a particular behavior (your dog barking).\n\nHuman: Huh? Am I supposed to wait until he barks and then say speak? Do I use treats?\n\nAssistant:']
input_sentences = [
    "This is a test",
    "One fish, two fish, red fish,",
    "Microsoft is in Washington",
    "The ancient art of basket weaving",
    "Large language models are useful",
    "You shouldn't buy a car without first checking",
    "In today's lesson, we will cover the conflict between",
    "Interestingly, the humble bumblebee is essential to our",
    "How many blue buttons and yellow marbles are left in the",
    "My favorite band is playing at a local music festival next month",
    "Fortunately, I made it just in time to the event to tell her that",
    "Once upon a time in a galaxy far away, there lived a boy named Anakin who",
    "It is projected that by the year 3035, there will be more humans on the planet than ",
    "Many years ago, we were hiking in the Amazon rain forest when we stumbled upon an impressive",
    "Let's discuss today's agenda. First, we will go around and introduce ourselves. Next, we will cover our 3 essential markers for success: 1) ",
    "These two historical figures ",
    "I saw a news article about a major scientific discovery ",
    "A poem about the beauty of the night sky",
    "Improving mental health ",
    "Being a professional athlete",
    "There are many exotic travel destinations",
    "She needed a recipe for a unique and delicious dessert",
    "The process of creating a work of art",
    "The importance of renewable energy has been a popular topic among",
    "Hiking to the top of a mountain is no easy task. It can takes several hours and ",
    "His latest clothing collection was all the rave at the last year's fashion week. Several ",
    "Here's a list of 10 thought-provoking discussion questions",
    "The show last night had to be postponed due to weather. I heard that people waited hours in the rain",
    "A successful small business can be evaluated these three performance metrics",
    "My favorite motivational quotes to inspire others are",
    "A magical creature living in a hidden forest",
    "The preparation of a gourmet meal",
    "I overheard two scientists discussing a groundbreaking scientific theory",
    "He wrote a blog post about the benefits of mindfulness and meditation.",
    "This set of instructions for assembling a piece of furniture",
    "Training for a marathon",
    "What are your hopes and dreams for the world?",
    "Imagine you are a time traveler. Write a journal entry about your visit to a historical event.",
    "Generate a list of 10 unique and exciting travel destinations.",
    "She gave speech advocating for equal rights",
    "The process of producing a documentary film ",
    "With a flick of a wand, the magician made the rabbit disappear",
    "The bustling marketplace was a kaleidoscope of colors and sounds. There were at least 100 vendors and dozens of"
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]

iters = 30 if args.test_performance else 2 #warmup
times = []
for i in range(iters):
    torch.cuda.synchronize()
    start = time.time()
    outputs = pipe(inputs,
            num_tokens=args.max_new_tokens,
            do_sample=(not args.greedy))
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)
print(f"generation time is {times[1]} sec")

if args.local_rank == 0:
    for i, o in zip(inputs, outputs):
        print(f"\nin={i}\nout={o}\n{'-'*60}")
    if args.test_performance:
        print_perf_stats(map(lambda t: t / args.max_new_tokens, times), pipe.model.config)

