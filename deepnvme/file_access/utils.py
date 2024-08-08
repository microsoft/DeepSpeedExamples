import os
import argparse
import torch
import timeit
import functools
import pathlib

def parse_read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        default=None,
                        type=str,
                        required=True,
                        help='File to read.')
    args = parser.parse_args()
    print(f'args = {args}')
    if not os.path.isfile(args.input_file):
        print(f'Invalid input file path: {args.input_file}')
        quit()

    return args



def parse_write_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder',
                        default=None,
                        type=str,
                        required=True,
                        help='Output folder for file write.')
    parser.add_argument('--mb_size',
                        type=int,
                        default=None,
                        required=True,
                        help='Size of tensor to save in MB.')
    
    args = parser.parse_args()
    print(f'args = {args}')
    if not os.path.isdir(args.output_folder):
        print(f'Invalid output folder path: {args.output_folder}')
        quit()

    return args

def create_file(file_path, file_sz):
    print(f'creating {file_path}')
    with open(file_path, 'wb') as f:
        f.write(os.urandom(file_sz))


def parse_nvme_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nvme_folder', 
                        default=None, 
                        type=str, 
                        required=True, 
                        help='Folder on NVMe device to use for file read/write.')
    parser.add_argument('--mb_size',
                        type=int,
                        default=None,
                        required=True,
                        help='I/O data size in MB.')
    parser.add_argument('--loop',
                        type=int,
                        default=3,
                        help='The number of times to repeat the operation.')

    args = parser.parse_args()
    print(f'{args=}')
    if not os.path.isdir(args.nvme_folder):
        print(f'Invalid folder path: {args.nvme_folder}')
        quit()

    args.file_path = os.path.join(args.nvme_folder, f'_deepnvme_file_data_{args.mb_size}MB.pt')

    return args 
