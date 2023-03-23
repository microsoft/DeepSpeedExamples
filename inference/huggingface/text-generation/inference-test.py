from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed
import math
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
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
        print(num_parameters)
        print(num_bytes)
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
#print(pipe.model)
if args.ds_inference:
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


input_sentences = [
         "DeepSpeed is a machine learning framework for deep neural network research that has gained widespread use among researchers ie pyDeepSpeed especially in recent years \
         While not the first such package for deep learning in R as many of us are familiar DeepSpeed has been around in some form since as early as 20142Fn2ref-type=nnThe recent \
         resurgence in popularity of DeepSpeed stems largely from the fact that it is easy to use has the full complement of modern deep learning library features and has a \
         well-organized community of contributors who are responsive knowledgeable and helpfulnnIf you have a question about the use of DeepSpeed the process for submitting a good \
         question and getting a fair response is as follows:nnFirst please use the existing GitHubGitHub user group to ask general questions This group is specifically for users of \
         DeepSpeed. DeepSpeed and the pyDeepSpeedpyDeepSpeed package If your question does not have enough detail to be specific it will probably be better suited to the GitHubGitHub \
         user group for general R usersnnIf you do have a valid question post it in the following sectionnnQuestions regarding the package itself its documentation features or other \
         functionality:nnAsk on GitHubGitHub Note that GitHub is the de-facto user group for questions that require more structure and context than you might find in \
         commentsnnGitHubGitHub for questions about the package itself specifically issues pull requests reporting bugs etcnnQuestions about the DeepSpeed user community itself or \
         questions relevant to the research community:nnDeepSpeed issues and forums are a good place to ask general questions that do not relate to a question about the functionality \
         of the code or packages but rather to questions regarding collaboration between DeepSpeed usersnnOther questions:nnPlease do not use this channel for other questions which \
         could be on-topic at another forum or the wrong place Asking questions in the Questions about DeepSpeed and related topics forum is a great way to get started with the \
         communitynn<https:wwwdeepspeedorgcommunitiesusershtml>nnIf you have a question regarding how to use the package functionality not addressed here please post a GitHub \
         issueGitHub or make a comment in the comments on this page If you are looking for feedback on a problem you have encountered please use the github Issues channel as detailed \
         abovennTips and general guidelines:nnIn the GitHub issueshttp:helpgithubcomtagref_issue channel use the category feature_request for reporting bugs or making feature requests; \
         use the category question for asking questions you have to get answers to or reporting general questions about the packagennIf you have issues or questions on a feature that \
         While not the first such package for deep learning in R as many of us are familiar DeepSpeed has been around in some form since as early as 20142Fn2ref-type=nnThe recent \
         resurgence in popularity of DeepSpeed stems largely from the fact that it is easy to use has the full complement of modern deep learning library features and has a \
         well-organized community of contributors who are responsive knowledgeable and helpfulnnIf you have a question about the use of DeepSpeed the process for submitting a good \
         question and getting a fair response is as follows:nnFirst please use the existing GitHubGitHub user group to ask general questions This group is specifically for users of \
         DeepSpeedDeepSpeed and the pyDeepSpeedpyDeepSpeed package If your question does not have enough detail to be specific it will probably be better suited to the GitHubGitHub \
         user group for general R usersnnIf you do have a valid question post it in the following sectionnnQuestions regarding the package itself its documentation features or other \
         functionality:nnAsk on GitHubGitHub Note that GitHub is the de-facto user group for questions that require more structure and context than you might find in \
         commentsnnGitHubGitHub for questions about the package itself specifically issues pull requests reporting bugs etcnnQuestions about the DeepSpeed user community itself or \
         questions relevant to the research community:nnDeepSpeed issues and forums are a good place to ask general questions that do not relate to a question about the functionality \
         of the code or packages but rather to questions regarding collaboration between DeepSpeed usersnnOther questions:nnPlease do not use this channel for other questions which \
         could be on-topic at another forum or the wrong place Asking questions in the Questions about DeepSpeed and related topics forum is a great way to get started with the \
         communitynn<https:wwwdeepspeedorgcommunitiesusershtml>nnIf you have a question regarding how to use the package functionality not addressed here please post a GitHub \
         issueGitHub or make a comment in the comments on this page If you are looking for feedback on a problem you have encountered please use the github Issues channel as detailed \
         abovennTips and general guidelines:nnIn the GitHub issueshttp:helpgithubcomtagref_issue channel use the category feature_request for reporting bugs or making feature requests; \
         use the category question for asking questions you have to get answers to or reporting general questions about the packagennIf you have issues or questions on a feature that \
         While not the first such package for deep learning in R as many of us are familiar DeepSpeed has been around in some form since as early as 20142Fn2ref-type=nnThe recent \
         resurgence in popularity of DeepSpeed stems largely from the fact that it is easy to use has the full complement of modern deep learning library features and has a \
         While not the first such package for deep learning in R as many of us are familiar DeepSpeed has been around in some form since as early as 20142Fn2ref-type=nnThe recent \
         resurgence in popularity of DeepSpeed stems largely from the fact that it is easy to use has the full complement of modern deep learning library features and has a \
         well-organized community of contributors who are responsive knowledgeable and helpfulnnIf you have a question about the use of DeepSpeed the process for submitting a good \
         question and getting a fair response is as follows:nnFirst please use the existing GitHubGitHub user group to ask general questions This group is specifically for users of \
         DeepSpeed. DeepSpeed and the pyDeepSpeedpyDeepSpeed package If your question does not have enough detail to be specific it will probably be better suited to the GitHubGitHub \
         user group for general R usersnnIf you do have a valid question post it in the following sectionnnQuestions regarding the package itself its documentation features or other \
         functionality:nnAsk on GitHubGitHub Note that GitHub is the de-facto user group for questions that require more structure and context than you might find in \
         commentsnnGitHubGitHub for questions about the package itself specifically issues pull requests reporting bugs etcnnQuestions about the DeepSpeed user community itself or \
         questions relevant to the research community:nnDeepSpeed issues and forums are a good place to ask general questions that do not relate to a question about the functionality \
         of the code or packages but rather to questions regarding collaboration between DeepSpeed usersnnOther questions:nnPlease do not use this channel for other questions which \
         could be on-topic at another forum or the wrong place Asking questions in the Questions about DeepSpeed and related topics forum is a great way to get started with the \
         communitynn<https:wwwdeepspeedorgcommunitiesusershtml>nnIf you have a question regarding how to use the package functionality not addressed here please post a GitHub \
         issueGitHub or make a comment in the comments on this page If you are looking for feedback on a problem you have encountered please use the github Issues channel as detailed \
         abovennTips and general guidelines:nnIn the GitHub issueshttp:helpgithubcomtagref_issue channel use the category feature_request for reporting bugs or making feature requests; \
         use the category question for asking questions you have to get answers to or reporting general questions about the packagennIf you have issues or questions on a feature that \
         While not the first such package for deep learning in R as many of us are familiar DeepSpeed has been around in some form since as early as 20142Fn2ref-type=nnThe recent \
         resurgence in popularity of DeepSpeed stems largely from the fact that it is easy to use has the full complement of modern deep learning library features and has a \
         well-organized community of contributors who are responsive knowledgeable and helpfulnnIf you have a question about the use of DeepSpeed the process for submitting a good \
         question and getting a fair response is as follows:nnFirst please use the existing GitHubGitHub user group to ask general questions This group is specifically for users of \
         DeepSpeedDeepSpeed and the pyDeepSpeedpyDeepSpeed package If your question does not have enough detail to be specific it will probably be better suited to the GitHubGitHub \
         user group for general R usersnnIf you do have a valid question post it in the following sectionnnQuestions regarding the package itself its documentation features or other \
         functionality:nnAsk on GitHubGitHub Note that GitHub is the de-facto user group for questions that require more structure and context than you might find in \
         commentsnnGitHubGitHub for questions about the package itself specifically issues pull requests reporting bugs etcnnQuestions about the DeepSpeed user community itself or \
         questions relevant to the research community:nnDeepSpeed issues and forums are a good place to ask general questions that do not relate to a question about the functionality \
         of the code or packages but rather to questions regarding collaboration between DeepSpeed usersnnOther questions:nnPlease do not use this channel for other questions which \
         could be on-topic at another forum or the wrong place Asking questions in the Questions about DeepSpeed and related topics forum is a great way to get started with the \
         communitynn<https:wwwdeepspeedorgcommunitiesusershtml>nnIf you have a question regarding how to use the package functionality not addressed here please post a GitHub \
         issueGitHub or make a comment in the comments on this page If you are looking for feedback on a problem you have encountered please use the github Issues channel as detailed \
         abovennTips and general guidelines:nnIn the GitHub issueshttp:helpgithubcomtagref_issue channel use the category feature_request for reporting bugs or making feature requests; \
         use the category question for asking questions you have to get answers to or reporting general questions about the packagennIf you have issues or questions on a feature that \
         While not the first such package for deep learning in R as many of us are familiar DeepSpeed has been around in some form since as early as 20142Fn2ref-type=nnThe recent \
         resurgence in popularity of DeepSpeed stems largely from the fact that it is easy to use has the full complement of modern deep learning library features and has a \
         While not the first such package for deep learning in R as many of us are familiar DeepSpeed has been around in some form since as early as 20142Fn2ref-type=nnThe recent \
         resurgence in popularity of DeepSpeed stems largely from the fact that it is easy to use has the full complement of modern deep learning library features and has a \
         well-organized community of contributors who are responsive knowledgeable and helpfulnnIf you have a question about the use of DeepSpeed the process for submitting a good \
         question and getting a fair response is as follows:nnFirst please use the existing GitHubGitHub user group to ask general questions This group is specifically for users of \
         DeepSpeed. DeepSpeed and the pyDeepSpeedpyDeepSpeed package If your question does not have enough detail to be specific it will probably be better suited to the GitHubGitHub \
         user group for general R usersnnIf you do have a valid question post it in the following sectionnnQuestions regarding the package itself its documentation features or other \
         functionality:nnAsk on GitHubGitHub Note that GitHub is the de-facto user group for questions that require more structure and context than you might find in \
         commentsnnGitHubGitHub for questions about the package itself specifically issues pull requests reporting bugs etcnnQuestions about the DeepSpeed user community itself or \
         questions relevant to the research community:nnDeepSpeed issues and forums are a good place to ask general questions that do not relate to a question about the functionality \
         of the code or packages but rather to questions regarding collaboration between DeepSpeed usersnnOther questions:nnPlease do not use this channel for other questions which \
         could be on-topic at another forum or the wrong place Asking questions in the Questions about DeepSpeed and related topics forum is a great way to get started with the \
         communitynn<https:wwwdeepspeedorgcommunitiesusershtml>nnIf you have a question regarding how to use the package functionality not addressed here please post a GitHub \
         issueGitHub or make a comment in the comments on this page If you are looking for feedback on a problem you have encountered please use the github Issues channel as detailed \
         abovennTips and general guidelines:nnIn the GitHub issueshttp:helpgithubcomtagref_issue channel use the category feature_request for reporting bugs or making feature requests; \
         use the category question for asking questions you have to get answers to or reporting general questions about the packagennIf you have issues or questions on a feature that \
         While not the first such package for deep learning in R as many of us are familiar DeepSpeed has been around in some form since as early as 20142Fn2ref-type=nnThe recent \
         resurgence in popularity of DeepSpeed stems largely from the fact that it is easy to use has the full complement of modern deep learning library features and has a \
         well-organized community of contributors who are responsive knowledgeable and helpfulnnIf you have a question about the use of DeepSpeed the process for submitting a good \
         question and getting a fair response is as follows:nnFirst please use the existing GitHubGitHub user group to ask general questions This group is specifically for users of \
         DeepSpeedDeepSpeed and the pyDeepSpeedpyDeepSpeed package If your question does not have enough detail to be specific it will probably be better suited to the GitHubGitHub \
         user group for general R usersnnIf you do have a valid question post it in the following sectionnnQuestions regarding the package itself its documentation features or other \
         functionality:nnAsk on GitHubGitHub Note that GitHub is the de-facto user group for questions that require more structure and context than you might find in \
         commentsnnGitHubGitHub for questions about the package itself specifically issues pull requests reporting bugs etcnnQuestions about the DeepSpeed user community itself or \
         questions relevant to the research community:nnDeepSpeed issues and forums are a good place to ask general questions that do not relate to a question about the functionality \
         of the code or packages but rather to questions regarding collaboration between DeepSpeed usersnnOther questions:nnPlease do not use this channel for other questions which \
         could be on-topic at another forum or the wrong place Asking questions in the Questions about DeepSpeed and related topics forum is a great way to get started with the \
         communitynn<https:wwwdeepspeedorgcommunitiesusershtml>nnIf you have a question regarding how to use the package functionality not addressed here please post a GitHub \
         issueGitHub or make a comment in the comments on this page If you are looking for feedback on a problem you have encountered please use the github Issues channel as detailed \
         abovennTips and general guidelines:nnIn the GitHub issueshttp:helpgithubcomtagref_issue channel use the category feature_request for reporting bugs or making feature requests; \
         use the category question for asking questions you have to get answers to or reporting general questions about the packagennIf you have issues or questions on a feature that \
         While not the first such package for deep learning in R as many of us are familiar DeepSpeed has been around in some form since as early as 20142Fn2ref-type=nnThe recent \
         resurgence in popularity of DeepSpeed stems largely from the fact that it is easy to use has the full complement of modern deep learning library features and has a \
         While not the first such package for deep learning in R as many of us are familiar DeepSpeed has been around in some form since as early as 20142Fn2ref-type=nnThe recent \
         resurgence in popularity of DeepSpeed stems largely from the fact that it is easy to use has the full complement of modern deep learning library features and has a \
         well-organized community of contributors who are responsive knowledgeable and helpfulnnIf you have a question about the use of DeepSpeed the process for submitting a good \
         question and getting a fair response is as follows:nnFirst please use the existing GitHubGitHub user group to ask general questions This group is specifically for users of \
         DeepSpeed. DeepSpeed and the pyDeepSpeedpyDeepSpeed package If your question does not have enough detail to be specific it will probably be better suited to the GitHubGitHub \
         user group for general R usersnnIf you do have a valid question post it in the following sectionnnQuestions regarding the package itself its documentation features or other \
         functionality:nnAsk on GitHubGitHub Note that GitHub is the de-facto user group for questions that require more structure and context than you might find in \
         commentsnnGitHubGitHub for questions about the package itself specifically issues pull requests reporting bugs etcnnQuestions about the DeepSpeed user community itself or \
         questions relevant to the research community:nnDeepSpeed issues and forums are a good place to ask general questions that do not relate to a question about the functionality \
         well-organized community of contributors who are responsive knowledgeable and helpfulnnIf you have a question about the use of DeepSpeed the process for submitting a good ",
         "DeepSpeed is a machine learning framework ",
         "He is working on",
         "He has a",
         "He got all",
         "Everyone is happy and I can",
         "The new movie that got Oscar this year",
         "In the far far distance from our galaxy,",
         "Peace is the only way"
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]

iters = 30 if args.test_performance else 1 #warmup
times = []
for i in range(iters):
    torch.cuda.synchronize()
    start = time.time()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    outputs = pipe(inputs,
            num_tokens=args.max_new_tokens,
            do_sample=(not args.greedy))
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)
#print(f"generation time is {times[1]} sec")

#print(prof.key_averages().table(sort_by="cuda_time_total"))#, row_limit=20))

if args.local_rank == 0:
    # for i, o in zip(inputs, outputs):
    #     print(f"\nin={i}\nout={o}\n{'-'*60}")
    if args.test_performance:
        print_perf_stats(map(lambda t: t / (args.max_new_tokens*args.batch_size), times), pipe.model.config)

