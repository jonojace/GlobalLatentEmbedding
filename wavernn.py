import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *
import sys
import models.nocond as nc
import models.vqvae as vqvae
import models.wavernn1 as wr
import utils.env as env
import argparse
import platform
import re
import utils.logger as logger
import time
import subprocess
from tensorboardX import SummaryWriter

import config

parser = argparse.ArgumentParser(description='Train or run some neural net')
parser.add_argument('--generate', '-g', action='store_true', help='run generation rather than training')
parser.add_argument('--float', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--load', '-l', help="provide full path to the model checkpoint to load")
parser.add_argument('--scratch', action='store_true')
parser.add_argument('--model', '-m', help='model type')
parser.add_argument('--force', action='store_true', help='skip the version check')
parser.add_argument('--test-speakers', type=int, default=3, help='number of speakers in the test set (if equal to 0, then uses all speakers)')
parser.add_argument('--test-utts-per-speaker', type=int, default=30, help='number of test utts from each test speaker (if equal to 0, then uses all utts from each test speaker)')
parser.add_argument('--partial', action='append', default=[], help='model to partially load')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help="initial learning rate")
parser.add_argument('--weight-decay', default=1e-04, type=float, help="weight decay (default: 1e-04)")
parser.add_argument('--batch-size', type=int, default=48, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--beta', type=float, default=0., help='the beta of singular loss')
parser.add_argument('--epochs', type=int, default=1000, help='epochs in training')
parser.add_argument('--test-epochs', type=int, default=200, help='testing every X epochs')
parser.add_argument('--num-group', type=int, default=8, help='num of groups in dictionary')
parser.add_argument('--num-sample', type=int, default=1, help='num of Monte Carlo samples')

# Introduced by Jason Fong
parser.add_argument('--only-gen-discrete', action='store_true', help='prevent decoder from running when generating from a trained model to speed up generation of discrete tokens')
parser.add_argument('--tokens-path', help="provide full path to the discrete tokens file to generate from")
parser.add_argument('--name', '-n', help='model identifier')

args = parser.parse_args()

if args.float and args.half:
    sys.exit('--float and --half cannot be specified together')

if args.float:
    use_half = False
elif args.half:
    use_half = True
else:
    use_half = False

model_type = args.model or 'vqvae'

if args.name:
    model_name = f'{model_type}.{args.name}'
else:
    model_name = f'{model_type}'

if model_type[:5] == 'vqvae':
    print("Model type is vqvae")
    model_fn = lambda dataset: vqvae.Model(model_type=model_type, rnn_dims=896, fc_dims=896, global_decoder_cond_dims=dataset.num_speakers(),
                  upsample_factors=(4, 4, 4), num_group=args.num_group, num_sample=args.num_sample, normalize_vq=True, noise_x=True, noise_y=True).cuda()
    dataset_type = 'multi'
elif model_type == 'wavernn':
    print("Model type is wavernn")
    model_fn = lambda dataset: wr.Model(rnn_dims=896, fc_dims=896, pad=2,
                  upsample_factors=(4, 4, 4), feat_dims=80).cuda()
    dataset_type = 'single'
elif model_type == 'nc':
    print("Model type is nc")
    model_fn = lambda dataset: nc.Model(rnn_dims=896, fc_dims=896).cuda()
    dataset_type = 'single'
else:
    sys.exit(f'Unknown model: {model_type}')

if dataset_type == 'multi':
    data_path = config.multi_speaker_data_path
    with open(f'{data_path}/index.pkl', 'rb') as f:
        index = pickle.load(f)

    logger.log(f"len of vctk index pkl object is {len(index)}") # should be equal to total number of speakers in the dataset
    # logger.log(f"index.pkl file --- index[:5] {index[:5]}")
    # logger.log(f"index.pkl file --- index[0][:5] {index[0][:5]}")

    test_index = [x[:args.test_utts_per_speaker] if i < args.test_speakers else [] for i, x in enumerate(index)] # take first 30 utts from args.test_speakers speakers as test data
    train_index = [x[args.test_utts_per_speaker:] if i < args.test_speakers else x for i, x in enumerate(index)] # rest of utts are training data from each speaker
    dataset = env.MultispeakerDataset(train_index, data_path)
elif dataset_type == 'single':
    data_path = config.single_speaker_data_path
    with open(f'{data_path}/dataset_ids.pkl', 'rb') as f:
        index = pickle.load(f)
    test_index = index[-args.test_speakers:] + index[:args.test_speakers]
    train_index = index[:-args.test_speakers]
    dataset = env.AudiobookDataset(train_index, data_path)
else:
    raise RuntimeError('bad dataset type')

print(f'dataset size: {len(dataset)}')

model = model_fn(dataset)

if use_half:
    model = model.half()

for partial_path in args.partial:
    model.load_state_dict(torch.load(partial_path), strict=False)

optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

paths = env.Paths(model_name, data_path)

if args.scratch or args.load == None and not os.path.exists(paths.model_path()):
    # Start from scratch
    step = 0
    epoch = 0
else:
    if args.load:
        #remove .pyt extension and step number
        prev_model_name = re.sub(r'_[0-9]+$', '', re.sub(r'\.pyt$', '', os.path.basename(args.load)))
        prev_model_basename = prev_model_name.split('_')[0]
        model_basename = model_name.split('_')[0]
        if prev_model_basename != model_basename and not args.force:
            sys.exit(f'refusing to load {args.load} because its prev_model_basename:({prev_model_basename}) is not model_basename:({model_basename})')
        if args.generate:
            paths = env.Paths(prev_model_name, data_path)
        prev_path = args.load
    else:
        prev_path = paths.model_path()
    step, epoch = env.restore(prev_path, model, optimiser)

#model.freeze_encoder()



if args.generate:
    if args.tokens_path:
        model.do_generate_from_tokens(paths,
                                      args.tokens_path,
                                      verbose=True)
    else:
        model.do_generate(paths,
                          data_path,
                          index,
                          args.test_speakers,
                          args.test_utts_per_speaker,
                          use_half=use_half,
                          verbose=True,
                          only_discrete=args.only_gen_discrete)
else:
    logger.set_logfile(paths.logfile_path())
    logger.log('------------------------------------------------------------')
    logger.log('-- New training session starts here ------------------------')
    logger.log(time.strftime('%c UTC', time.gmtime()))
    logger.log('beta={}'.format(args.beta))
    logger.log('num_group={}'.format(args.num_group))
    logger.log('test_speakers ={}'.format(args.test_speakers))
    logger.log('num_sample={}'.format(args.num_sample))
    writer = SummaryWriter(paths.logfile_path() + '_tensorboard')
    writer.add_scalars('Params/Train', {'beta': args.beta})
    writer.add_scalars('Params/Train', {'num_group': args.num_group})
    writer.add_scalars('Params/Train', {'num_sample': args.num_sample})
    #model.do_train(paths, dataset, optimiser, writer, epochs=args.epochs, test_epochs=args.test_epochs, batch_size=args.batch_size, step=step, epoch=epoch, use_half=use_half, valid_index=test_index, beta=args.beta)
    if model_type[:5] == 'vqvae':
        model.do_train(paths,
                       dataset,
                       optimiser,
                       writer,
                       epochs=args.epochs,
                       test_epochs=args.test_epochs,
                       batch_size=args.batch_size,
                       step=step,
                       epoch=epoch,
                       use_half=use_half,
                       valid_index=test_index,
                       beta=args.beta)
    elif model_type == 'wavernn':
        model.do_train(paths,
                       dataset,
                       optimiser,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       step=step,
                       valid_index=test_index,
                       use_half=use_half)
    else:
        sys.exit(f'Unknown model: {model_type}')

