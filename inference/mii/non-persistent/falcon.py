from mii import pipeline

pipe = pipeline("tiiuae/falcon-7b")
responses = pipe("DeepSpeed is", max_new_tokens=128, return_full_text=True)
if pipe.is_rank_0:
    print(responses[0])
