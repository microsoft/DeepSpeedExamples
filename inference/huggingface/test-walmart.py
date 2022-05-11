from transformers import pipeline
import deepspeed
import transformers
import torch
import datetime
import os
import time
from transformers import AutoModel, AutoTokenizer
def run_model(name , enable_cuda_graph, batch, seq):
    print("Loading model...")
    #name = 'distilbert-base-uncased'
#    generator = pipeline('fill-mask', model = 'bert-large-cased', device = 0)[0]
    model = AutoModel.from_pretrained(name)
    model = model.eval()


    #exit()
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokens = tokenizer(
        '''
		Hello I'm a [MASK] researcher. I work with very big and great [MASK].
		Hello I'm a [MASK] researcher. I work with very big and great [MASK].
		Hello I'm a [MASK] researcher. I work with very big and great [MASK].
		Hello I'm a [MASK] researcher. I work with very big and great [MASK].
		Hello I'm a [MASK] researcher. I work with very big and great [MASK].
		Hello I'm a [MASK] researcher. I work with very big and great [MASK].
		Hello I'm a [MASK] researcher. I work with very big and great [MASK].
		Hello I'm a [MASK].
	''',
        return_tensors="pt",
    )
    tokens['input_ids'] = torch.cuda.LongTensor([[1356]*seq]*batch)#torch.empty(1,19,dtype=torch.long)
    tokens['attention_mask'] = torch.cuda.LongTensor([[1]*seq]*batch)#torch.empty(1,19,dtype=torch.long)
    model = model.half().cuda()
    print(f" #tokens is {tokens['input_ids'].shape[1]}")
#    exit()
    for t in tokens:
            if torch.is_tensor(tokens[t]):
                tokens[t] = tokens[t].cuda()
    #print(f"baseline{model(**tokens)}")
    model = deepspeed.init_inference(model,
                                      dtype=torch.int8,
                                      replace_with_kernel_inject=True,
                                      enable_qkv_quantization=True,
                                      #enable_cuda_graph=(enable_cuda_graph=='1')
					)
#    print(model(**tokens))
#    print(model(**tokens))
    ret = model(**tokens)
    #print(ret)
    #ret1 = torch.empty_like(ret)
    #ret2 = torch.empty_like(ret1)
    if False: #enable_cuda_graph=='1'
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for i in range(3):
                    ret = model(**tokens) #generator("Hello I'm a [MASK] researcher.")
            
            torch.cuda.current_stream().wait_stream(s)
            #print(ret)
            #exit()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                ret1 = model(**tokens) #generator("Hello I'm a [MASK] researcher.")
    #print(ret1)
    #exit()
    #ret1.copy_(ret2)
    torch.cuda.synchronize()
    t1 = time.time() #datetime.datetime.now()
    #for t in tokens:
    #        if torch.is_tensor(tokens[t]):
    #            tokens[t].mul_(2)
    
    for _ in range(100):
        #print(tokens)
        model(**tokens) #generator("Hello I'm a [MASK] researcher.")
        #print(ret1)
    torch.cuda.synchronize()
    t2 = time.time()#datetime.datetime.now()
    print(str((t2 - t1) * 10) + " ms")
import sys
if __name__ == '__main__':
    run_model(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

