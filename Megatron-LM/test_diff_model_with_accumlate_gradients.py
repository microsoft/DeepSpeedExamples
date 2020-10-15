import glob
import os
import re
import numpy as np
import os.path
from os import path

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

result_dir = "result_with_acumulated_grad/"
if path.exists(result_dir) is False:
    os.mkdir(result_dir)


files = glob.glob("model_size_test_with_acumulated_grad/ds_zero-offload_*_pretrain_gpt2_model_parallel.sh")

for f in files:
    output_file = result_dir+"%s_result.txt"%find_between( f, "ds_zero-offload_", "_pretrain" )
    print(output_file)
    exe_commamd = "bash " +f+ " > %s"%output_file
    print("executing %s"% exe_commamd)
    os.system(exe_commamd)
    print("================")