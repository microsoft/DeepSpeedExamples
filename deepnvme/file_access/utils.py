import os
import argparse


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
