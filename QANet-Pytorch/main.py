# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
import os,sys
import argparse
from datetime import datetime
from model_train.ema import EMA
from model_train.visualize import Visualizer
from data_process import Data_process, pickle_load_large_file
from SQuADdataset import SQuADDataset, collate
from torch.utils.data import DataLoader
from QANet_Unanswerable import QANet_Unanswerable
from model_train.model_train import Model_Trainer

def add_argument():

    parser=argparse.ArgumentParser(description='Implementing the QANet to deal with unanswerable questions')

    #data
    parser.add_argument('--data_processed', default=False, action='store_true',
                        help='whether the data has been processed')
    parser.add_argument('--train_data', default='data/original/SQuAD/train-v1.1.json', type=str,
                        help='path of train data')
    parser.add_argument('--dev_data', default='data/original/SQuAD/dev-v1.1.json', type=str,
                        help='path of dev data')
    parser.add_argument('--train_processed_data', default='data/processed/SQuAD/train-v1.1-processed.pkl', type=str,
                        help='path of train processed data')
    parser.add_argument('--dev_processed_data', default='data/processed/SQuAD/dev-v1.1-processed.pkl', type=str,
                        help='path of dev processed data')
    parser.add_argument('--train_meta_data', default='data/processed/SQuAD/train-v1.1-meta.pkl', type=str,
                        help='path of train meta data')
    parser.add_argument('--dev_meta_data', default='data/processed/SQuAD/dev-v1.1-meta.pkl', type=str,
                        help='path of dev meta data')
    parser.add_argument('--train_eval_data', default='data/processed/SQuAD/train-v1.1-eval.pkl', type=str,
                        help='path of train eval data')
    parser.add_argument('--dev_eval_data', default='data/processed/SQuAD/dev-v1.1-eval.pkl', type=str,
                        help='path of dev eval data')
    parser.add_argument('--eval_num_batches', default=500, type=int,
                        help='number of batches for evaluation')

    #word embedding
    parser.add_argument('--glove_word_embedding', default='data/original/Glove/glove.840B.300d.txt', type=str,
                        help='path of glove word embedding')
    parser.add_argument('--glove_word_size', default=int(2.2e6), type=int,
                        help='corpus size of glove word embedding')
    parser.add_argument('--glove_dim', default=300, type=int,
                        help='glove word embedding size (default: 300)')
    parser.add_argument('--processed_word_embedding', default='data/processed/SQuAD/word_emb.pkl', type=str,
                        help='path of word embedding matrix')
    parser.add_argument('--word_dictionary', default='data/processed/SQuAD/word_dic.pkl', type=str,
                        help='path of word embedding dictionary')
    #char embedding
    parser.add_argument('--char_emb_pretrained', default=False, action='store_true',
                        help='whether train char embedding or not')
    parser.add_argument('--glove_char_embedding', default='data/original/Glove/glove.840B.300d-char.txt', type=str,
                        help='path of glove char embedding')
    parser.add_argument('--glove_char_size', default=94, type=int,
                        help='corpus size of glove char embedding')
    parser.add_argument('--pretrained_char_emb_dim', default=64, type=int,
                        help='glove char embedding size (default: 200)')
    parser.add_argument('--processed_char_embedding', default='data/processed/SQuAD/char_emb.pkl', type=str,
                        help='path of char embedding matrix')
    parser.add_argument('--char_dictionary', default='data/processed/SQuAD/char_dic.pkl', type=str,
                        help='path of char embedding dictionary')

    #model
    parser.add_argument('--context_limit', default=400, type=int,
                        help='maximum number of context tokens')
    parser.add_argument('--question_limit', default=50, type=int,
                        help='maximum number of question tokens')
    parser.add_argument('--answer_limit', default=30, type=int,
                        help='maximum number of answer tokens')
    parser.add_argument('--char_limit', default=16, type=int,
                        help='maximum number of chars in a word')
    parser.add_argument('--model_dim', default=128, type=int,
                        help='model hidden state dimension')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='number of attention heads')

    # train
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')

    # debug
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug mode or not')
    parser.add_argument('--debug_batchnum', default=2, type=int,
                        help='only train and test a few batches when debug (devault: 2)')
    # checkpoint
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--verbosity', default=2, type=int,
                        help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
    parser.add_argument('--save_dir', default='checkpoints/', type=str,
                        help='directory of saved model (default: checkpoints/)')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='training checkpoint frequency (default: 1 epoch)')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='print training information frequency (default: 10 steps)')

    # cuda
    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--multi_gpu', default=False, action='store_true',
                        help='use multi-GPU in case there\'s multiple GPUs available')

    # log & visualize
    parser.add_argument('--visualizer',default=False, action='store_true',
                        help='use visdom visualizer or not')
    parser.add_argument('--log_file', default=None,type=str,
                        help='path of log file')

    # optimizer & scheduler & weight & exponential moving average
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--lr_warm_up_num', default=1000, type=int,
                        help='number of warm-up steps of learning rate')
    parser.add_argument('--beta1', default=0.8, type=float,
                        help='beta 1')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='beta 2')
    parser.add_argument('--decay', default=0.9999, type=float,
                        help='exponential moving average decay')
    parser.add_argument('--use_scheduler', default=True, action='store_false',
                        help='whether use learning rate scheduler')
    parser.add_argument('--use_grad_clip', default=True, action='store_false',
                        help='whether use gradient clip')
    parser.add_argument('--grad_clip', default=5.0, type=float,
                        help='global Norm gradient clipping rate')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('--use_early_stop', default=True, action='store_false',
                        help='whether use early stop')
    parser.add_argument('--early_stop', default=10, type=int,
                        help='checkpoints for early stop')

    args, _ =parser.parse_known_args()

    return args

def main(args):
    #show all arguments and configuration
    print(args)
    seed = 12345
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    #set log file
    log = sys.stdout
    if args.log_file is not None:
        log = open(args.log_file,'a')

    #set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    number_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print('device is cuda and cuda number is ', number_gpu)
    else:
        print('device is cpu')

    #process data
    if not args.data_processed:
        process_data = Data_process(args)
        process_data.process()

    #load word, char embedding and word dictionary
    word_emb_tensor = torch.FloatTensor(np.array(pickle_load_large_file(args.processed_word_embedding),
                                                 dtype=np.float32))
    char_emb_tensor = torch.FloatTensor(np.array(pickle_load_large_file(args.processed_char_embedding),
                                                 dtype=np.float32))
    word2idx_dict = pickle_load_large_file(args.word_dictionary)

    SQuAD_train_dataset = SQuADDataset(args.train_processed_data)
    train_data_loader = DataLoader(dataset=SQuAD_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                   collate_fn=collate)
    SQuAD_dev_dataset = SQuADDataset(args.dev_processed_data)
    dev_data_loader = DataLoader(dataset=SQuAD_dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                 collate_fn=collate)

    #initialize model
    model = QANet_Unanswerable(word_emb_tensor, char_emb_tensor, args.model_dim,
                        num_heads = args.num_heads, train_char_emb = args.char_emb_pretrained,
                        pad=word2idx_dict['<PAD>'])
    model.summary()
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    # exponential moving average
    ema = EMA(args.decay)
    if args.use_ema:
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

    # set optimizer and scheduler
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params = parameters, lr = args.lr, betas = (args.beta1, args.beta2),eps = 1e-8, weight_decay = 3e-7)
    cr = 1.0 / math.log(args.lr_warm_up_num)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda ee: cr * math.log(ee + 1) if ee < args.lr_warm_up_num else 1)

    # set loss, metrics
    loss = torch.nn.CrossEntropyLoss()

    # set visdom visualizer to store training process information
    # see the training process on http://localhost:8097/
    vis = None
    if args.visualizer:
        os.system("python -m visdom.server")
        vis = Visualizer("main")

    # construct trainer
    # an identifier (prefix) for saved model
    identifier = type(model).__name__ + '_'
    trainer = Model_Trainer(
        args, model, loss,
        train_data_loader=train_data_loader,
        dev_data_loader=dev_data_loader,
        dev_eval_file=args.dev_eval_data,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        with_cuda=args.with_cuda,
        save_dir=args.save_dir,
        verbosity=args.verbosity,
        save_freq=args.save_freq,
        print_freq=args.print_freq,
        resume=args.resume,
        identifier=identifier,
        debug=args.debug,
        debug_batchnum=args.debug_batchnum,
        lr=args.lr,
        lr_warm_up_num=args.lr_warm_up_num,
        grad_clip=args.grad_clip,
        decay=args.decay,
        visualizer=vis,
        logger=log,
        use_scheduler=args.use_scheduler,
        use_grad_clip=args.use_grad_clip,
        use_ema=args.use_ema,
        ema=ema,
        use_early_stop=args.use_early_stop,
        early_stop=args.early_stop)

    # start training!
    start = datetime.now()
    trainer.train()
    print("Time of training model ", datetime.now() - start)


if __name__ == '__main__':
    args=add_argument()
    main(args)
