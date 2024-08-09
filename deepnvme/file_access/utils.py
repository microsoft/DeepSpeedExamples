import os
import argparse

def parse_read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        default=None,
                        type=str,
                        required=True,
                        help='File to read, must be on NVMe device.')
    parser.add_argument('--loop',
                        type=int,
                        default=3,
                        help='The number of times to repeat the operation (default 3).')
    parser.add_argument('--validate',
                        action="store_true",
                        help="Run validation step that compares tensor value against Python file read")
    
    args = parser.parse_args()
    print(f'args = {args}')
    if not os.path.isfile(args.input_file):
        print(f'Invalid input file path: {args.input_file}')
        quit()

    return args



def parse_write_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nvme_folder',
                        default=None,
                        type=str,
                        required=True,
                        help='NVMe folder for file write.')
    parser.add_argument('--mb_size',
                        type=int,
                        default=1024,
                        help='Size of tensor to save in MB (default 1024).')   
    parser.add_argument('--loop',
                        type=int,
                        default=3,
                        help='The number of times to repeat the operation (default 3).')
    parser.add_argument('--validate',
                        action="store_true",
                        help="Run validation step that compares tensor value against Python file read")

    args = parser.parse_args()
    print(f'args = {args}')
    if not os.path.isdir(args.nvme_folder):
        print(f'Invalid output folder path: {args.output_folder}')
        quit()

    return args

