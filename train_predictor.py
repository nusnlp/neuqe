import os, sys
import argparse
import logging
import time
import shutil

import random
import torch
import torch.nn as nn
from torch import optim

import numpy as np

from neuqe.utils import log_utils as L
from neuqe.trainers import predictor_trainer as trainer
from neuqe.trainers import trainer_utils
from neuqe.models import model_utils
from neuqe.io import io_utils
from neuqe.io.vocab_manager import VocabManager
from neuqe.io.parallel_reader import ParallelReader
logger = logging.getLogger(__name__)



#########################
## MAIN TRAINING FUNCTION
#########################
def train(model, args, trainset_reader, vocab, validset_reader=None):

    debug=args.debug

    # for logging
    total_loss_value = 0

    #setting optimizers
    optimizer = trainer_utils.set_optimizer(args.optimizer)(model.parameters(), lr=args.learning_rate)

    #setting loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    trainset_reader.reset()
    num_batches = None
    best_valid_loss = None
    best_model = None
    is_best = False
    for epoch_idx in range(1,args.num_epochs+1):

        # shuffling trainset
        logger.info("shuffling batches...")
        random.seed(args.seed + (epoch_idx-1))
        if trainset_reader.shuffle_batches:
            trainset_reader.shuffle()

        # initializing minibatch
        minibatch_idx = 0
        minibatch = trainset_reader.next()

        while(minibatch):
            minibatch_idx += 1
            train_input = io_utils.create_predictor_input(minibatch,vocab)
            loss_value = trainer.train_step(train_input, model,optimizer, criterion, clip_norm=args.clip_norm, debug=debug)

            # calculating total loss for logging (per epoch)
            total_loss_value += loss_value

            # logging after set interval
            if minibatch_idx % args.log_interval == 0:
                trainer_utils.log_train_info(epoch_idx, minibatch_idx, total_loss_value, num_batches)

            if(debug==True):
                return

            # read next batch
            minibatch = trainset_reader.next()

        num_batches = minibatch_idx
        trainer_utils.log_train_info(epoch_idx, minibatch_idx, total_loss_value, num_batches)

        logger.info("epoch {} completed.".format(epoch_idx))
        total_loss_value = 0

        # validation
        if validset_reader:
            valid_loss = trainer.run_validation(model, validset_reader, vocab, debug=debug)
            is_best = False
            if best_valid_loss is None or best_valid_loss > valid_loss:
                best_epoch_idx = epoch_idx
                best_valid_loss = valid_loss
                is_best = True

            logger.info('validation: average loss per batch = %.4f (best %.4f @ epoch %d)'
                  % (valid_loss, best_valid_loss, best_epoch_idx))


            state = {
                'epoch': epoch_idx,
                'vocab':vocab,
                'args':args,
                'state_dict': model.state_dict(),
                'best_valid_loss': best_valid_loss,
                'best_epoch_idx': best_epoch_idx,
                'optimizer' : optimizer.state_dict(),
            }
            model_path = args.output_dir + '/model.epoch' + str(epoch_idx) + '.pt'
            best_model_path = args.output_dir + '/model.best.pt'
            trainer_utils.save_checkpoint(state, args.save_after_epochs, is_best, model_path=model_path, best_model_path=best_model_path)

##############
## Arguments
###############

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--train-prefix', required=True,  help='prefix of path to training dataset (without extension suffix).')
parser.add_argument('-valid', '--valid-prefix',  help='prefix of path to validation dataset (without extension suffix).')
parser.add_argument('-ssuf', '--source-suffix',  required=True, help='extension suffix of source part.')
parser.add_argument('-tsuf', '--target-suffix', required=True,  help='extension suffix of target part.')
parser.add_argument('-svocab', '--source-vocab-path', help='path to source vocab file (computed if not specified).')
parser.add_argument('-tvocab', '--target-vocab-path', help='path to target vocab file (computed if not specified).')
parser.add_argument('-saveall', '--save-after-epochs', action='store_true', help='flag to enable saving after every epoch (default: save the best model only)')
parser.add_argument('-nosave', '--no-save-best', action='store_true', help='flag to disable saving best model.')
parser.add_argument('-outdir', '--output-dir', required=True, help='path to output directory')
parser.add_argument('-arch', '--architecture', default='postech', help='architecture of predictor network ( postech | ctxpostech | convseq )')

## model_params
parser.add_argument('-nsvocab', '--num-source-vocab', default=30000, type=int, help='number of unique tokens in source vocab.')
parser.add_argument('-ntvocab', '--num-target-vocab', default=30000, type=int, help='number of unique tokens in target vocab.')
parser.add_argument('-maxslen', '--max-source-length', type=int, default=90, help='maximum length of source sentences to keep.')
parser.add_argument('-maxtlen', '--max-target-length', type=int, default=90, help='maximum length of target sentences to keep.')
parser.add_argument('-bsize','--batchsize', type=int, default=32, help='batch size for training')
parser.add_argument('-validbsize','--valid-batchsize', type=int, default=32, help='batch size for validation on validation set')
parser.add_argument('-nhid','--num-hidden-units', type=int, default=500, help='size of hidden units')
parser.add_argument('-nsembed','--num-source-embed', type=int, default=300, help='number of source embeddings'  )
parser.add_argument('-ntembed','--num-target-embed', type=int, default=300, help='number of target embeddings'  )
parser.add_argument('-nmaxout', '--num-maxout-units', type=int,  help='number of maxout units (default: nhid).')
parser.add_argument('-noutembed', '--num-output-embed', type=int,  help='number of output embeddings (default: nhid)')
parser.add_argument('-skwidth','--source-kernel-width', type=int, default=3, help='kernel width for convolutions (default: 3)')
parser.add_argument('-tkwidth','--target-kernel-width', type=int, default=3, help='kernel width for convolutions (default: 3)')
parser.add_argument('-nslayers','--num-source-layers', type=int, default=1, help='number of layers for convolutional models (default: 1)')
parser.add_argument('-ntlayers','--num-target-layers', type=int, default=1, help='number of layers for convolutional models (default: 1)')

## training parameters
parser.add_argument('-cnorm', '--clip-norm', default=None, type=float, help='clip value to clip gradients by L2 norm')
parser.add_argument('-nepochs', '--num-epochs', default=100, type=int, help='number of epochs to train.')
parser.add_argument('-opt', '--optimizer', default='adadelta', help='set the optimizer (adadelta|adam|adagrad|rmsprop|sgd)')
parser.add_argument('-lrate', '--learning-rate', default=1.0, type=float, help='learning rate')
parser.add_argument('-logafter', '--log-interval', default=1000, type=int, help='logging interval (in number of minibatches trained)')
parser.add_argument('-seed',  type=int, default=1234, help='seed to initialize randomness.')
parser.add_argument('-debug', action='store_true', help='flag to enable debugging by training only single batch')
args = parser.parse_args()


# create output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# initialize logger
handlers = [logging.FileHandler(os.path.abspath(args.output_dir)+'/train_log.txt'), logging.StreamHandler()]
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG, datefmt='%d-%m-%Y %H:%M:%S', handlers = handlers)

logger.info(args)


##############
## Setting up
##############
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# setting dataset paths
src_trainset_path = args.train_prefix + '.' + args.source_suffix
trg_trainset_path = args.train_prefix + '.' + args.target_suffix


# setting defaults of args.nmaxout and args.noembed to number of hidden units
if not args.num_maxout_units:
    args.num_maxout_units = args.num_hidden_units
if not args.num_output_embed:
    args.num_output_embed = args.num_hidden_units

# creating vocabs if necessary
if not args.source_vocab_path:
    args.source_vocab_path = args.output_dir+'/train.src.vocab'
    logger.info("writing source vocabulary to " + args.source_vocab_path)
    io_utils.write_vocab(src_trainset_path, args.source_vocab_path)
if not args.target_vocab_path:
    args.target_vocab_path = args.output_dir+'/train.trg.vocab'
    logger.info("writing target vocabulary to " + args.target_vocab_path)
    io_utils.write_vocab(trg_trainset_path, args.target_vocab_path)

logger.info("loading vocabularies")
src_vocab = VocabManager(vocab_path=args.source_vocab_path,num_vocab=args.num_source_vocab)
args.source_vocab_size = src_vocab.vocab_size # differs from num_vocab as it includes pad, unk, </s> etc.
trg_vocab = VocabManager(vocab_path=args.target_vocab_path,num_vocab=args.num_target_vocab)
args.target_vocab_size = trg_vocab.vocab_size

logger.info("loading training set")
trainset_reader = ParallelReader(source_dataset_path=src_trainset_path,
                                    target_dataset_path=trg_trainset_path,
                                    num_batches_in_cache=None,
                                    source_max_length=args.max_source_length,
                                    target_max_length=args.max_target_length,
                                    batchsize=args.batchsize,
                                    shuffle_batches=True)
validset_reader = None
if args.valid_prefix:
    logger.info("loading validation set")
    src_validset_path = args.valid_prefix + '.' + args.source_suffix
    trg_validset_path = args.valid_prefix + '.' + args.target_suffix
    validset_reader = ParallelReader(source_dataset_path=src_validset_path,
                                        target_dataset_path=trg_validset_path,
                                        num_batches_in_cache=None,
                                        source_max_length=args.max_source_length,
                                        target_max_length=args.max_target_length,
                                        batchsize=args.valid_batchsize,
                                        shuffle_batches=False)

# initialize model
Predictor = model_utils.set_predictor_arch(args.architecture)

logger.info("creating model")
model = Predictor(args).cuda()
total_params = sum(p.numel() for p in model.parameters())
logger.info("total number of parameters of the model: {}".format(total_params))

logger.info("starting training")
trainset_reader.reset()
vocab = (src_vocab,trg_vocab)
train(model, args, trainset_reader, vocab, validset_reader)
