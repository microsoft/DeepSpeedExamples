Accelerate, EleutherAI/gpt-j-6B, max_new_tokens=50, num_gpus=4
Batch size	1	4	8	16	32	64	128	200	244
Per Token Latency (ms)	49.65	57.81	60.86	66.79	71.95	116.03	206.28	330.8	371.15
Per GPU Throughput (Tflops)	0.0575	0.195	0.37	0.675	1.2525	1.555	1.75	1.705	1.8525

AutoTP, EleutherAI/gpt-j-6B, max_new_tokens=50, num_gpus=4
Batch size	1	4	8	16	32	64	128	200	244	512	768	1024	1280
Per Token Latency (ms)	50.7	48.63	49.67	47.63	50.89	50.72	55.89	52.26	63.33	80.05	110.1	130.87	148.6
Per GPU Throughput (Tflops)	0.055	0.2325	0.32	0.9475	1.7725	3.5575	6.455	21.57	10.86	18.0275	19.66	22.055	24.2775

