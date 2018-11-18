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
from neuqe.trainers import predictor_trainer, estimator_trainer
from neuqe.trainers import trainer_utils
from neuqe.models import model_utils
from neuqe.io import io_utils
from neuqe.io.vocab_manager import VocabManager
from neuqe.io.parallel_reader import ParallelReader
from neuqe.io.qualitydata_reader import QualityDataReader
logger = logging.getLogger(__name__)



#########################
## MAIN TRAINING FUNCTION
#########################
def train(est_model, pred_model, args, trainset_reader, vocab, validset_reader, testset_readers=None):
    """ Training function """
    debug=args.debug

    # for logging
    total_loss_value = 0

    #setting optimizers
    est_optimizer = trainer_utils.set_optimizer(args.optimizer)(filter(lambda p: p.requires_grad, est_model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    #setting loss function
    est_criterion = trainer_utils.set_criterion(args.loss)

    trainset_reader.reset()
    num_batches = None
    best_valid_loss = None
    best_model = None

    patience = 0
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

            # split into predictor input and estimator target scores
            pred_minibatch = [(src,hyp) for src,hyp,score in minibatch]
            scores = [score for src,hyp,score in minibatch]

            # create input as source, hypothesis pairs and their masks indexed with vocab
            train_input = io_utils.create_predictor_input(pred_minibatch,vocab)

            # perform a step of trainining
            loss_value = estimator_trainer.train_step(train_input, scores, est_model, est_optimizer, est_criterion, clip_norm=args.clip_norm, debug=args.debug)

            # calculating total loss for logging (per epoch)
            total_loss_value += loss_value

            # logging after set interval
            if minibatch_idx % args.log_interval == 0:
                trainer_utils.log_train_info(epoch_idx, minibatch_idx, total_loss_value, num_batches)
            if(debug==True):
                return

            # read next batch
            minibatch = trainset_reader.next()

        # find total number of batches
        num_batches = minibatch_idx

        # print the training log
        trainer_utils.log_train_info(epoch_idx, minibatch_idx, total_loss_value, num_batches)

        # completing one epoch
        logger.info("epoch {} completed.".format(epoch_idx))
        total_loss_value = 0

        #################
        # validation
        #################
        valid_loss, metric_scores = estimator_trainer.run_validation(est_model, validset_reader, vocab, est_criterion, metrics=args.metrics, debug=debug)

        is_best = False
        patience += 1
        if best_valid_loss is None or best_valid_loss > valid_loss:
            best_epoch_idx = epoch_idx
            best_valid_loss = valid_loss
            is_best = True
            patience = 0


        logger.info('epoch {0} validation \t\t| average {1} loss/batch = {2:.4f} (best {3:.4f} @ epoch {4})'.format(epoch_idx, args.loss, valid_loss, best_valid_loss, best_epoch_idx))
        if metric_scores:
            logger.info('epoch {0} validation \t\t| '.format(epoch_idx) + ', '.join(["{0}={1:.4f}".format(metric,score) for metric,score in metric_scores.items()]))

        state = {
            'epoch': epoch_idx,
            'args':args,
            'state_dict': est_model.state_dict(),
            'best_valid_loss': best_valid_loss,
            'best_epoch_idx': best_epoch_idx,
            'optimizer' : est_optimizer.state_dict(),
        }

        ##############
        # testing
        ##############
        if (testset_readers):
            for testset_reader in testset_readers:
                test_loss, metric_scores = estimator_trainer.run_validation(est_model, testset_reader, vocab, est_criterion, metrics=args.metrics, debug=debug)
                logger.info('epoch {0} testing on {1} \t\t| average {2} loss/batch = {3:.4f}'.format(epoch_idx,testset_reader.source_dataset_path,args.loss, test_loss))
                if metric_scores:
                    logger.info('epoch {0} testing on {1} \t\t| '.format(epoch_idx,testset_reader.source_dataset_path) + ', '.join(["{0}={1:.4f}".format(metric,score) for metric,score in metric_scores.items()]))

        ## saving the model
        est_model_path = args.output_dir + '/est_model.epoch' + str(epoch_idx) + '.pt'
        est_best_model_path = args.output_dir + '/est_model.best.pt'
        logger.info("saving model...")
        trainer_utils.save_checkpoint(state, args.save_after_epochs, is_best, args.no_save_best, est_model_path, est_best_model_path)

        if (patience >= args.patience):
            logger.info("early stopping at epoch {} (patience param: {})".format(epoch_idx, args.patience))
            logger.info("training complete.")
            break

##############
## Arguments
###############

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--train-prefix', required=True,  help='prefix of path to training dataset (without extension suffix).')
parser.add_argument('-valid', '--valid-prefix', required=True, help='prefix of path to validation dataset (without extension suffix).')
parser.add_argument('-test', '--test-prefixes', nargs='+',  help='space-separated prefixes of paths to test dataset (without extension suffix).')
parser.add_argument('-ssuf', '--source-suffix',  required=True, help='extension suffix of source part.')
parser.add_argument('-hsuf', '--hypothesis-suffix', required=True,  help='extension suffix of hypothesis part.')
parser.add_argument('-scoresuf', '--scores-suffix', required=True,  help='extension suffix of scores file.')
parser.add_argument('-pmodel','--predictor-model', required=False, help='path to trained predictor model.')
parser.add_argument('-qvectype', '--quality-vector-type', default='pre', help='type of quality vector (pre|post|prepost)')
parser.add_argument('-saveall', '--save-after-epochs', action='store_true', help='flag to enable saving after every epoch (default: save the best model only)')
parser.add_argument('-nosave', '--no-save-best', action='store_true', help='flag to disable saving best model.')
parser.add_argument('-outdir', '--output-dir', required=True, help='path to output directory')
parser.add_argument('-arch', '--architecture', default='postech', help='architecture of estimator network (postech|convseq)')

## model_params
parser.add_argument('-nsvocab', '--num-source-vocab', default=30000, type=int, help='number of unique tokens in source vocab.')
parser.add_argument('-ntvocab', '--num-target-vocab', default=30000, type=int, help='number of unique tokens in target vocab.')
parser.add_argument('-maxslen', '--max-source-length', type=int, default=90, help='maximum length of source sentences to keep.')
parser.add_argument('-maxhlen', '--max-hypothesis-length', type=int, default=90, help='maximum length of hypothesis sentences to keep.')
parser.add_argument('-bsize','--batchsize', type=int, default=32, help='batch size for training')
parser.add_argument('-validbsize','--valid-batchsize', type=int, default=32, help='batch size for validation on validation set')
parser.add_argument('-testbsize','--test-batchsize', type=int, default=1, help='batch size for validation on test set(s)')
parser.add_argument('-nhid','--num-hidden-units', type=int, default=100, help='size of hidden units')
parser.add_argument('-loss', default='mse', help='loss function to optimize (mse|pcorrel|mae)')
parser.add_argument('-metrics', nargs='+', help='space separated metrics to evaluate on, e.g pc,mae,mse (requires scikit and sklearn)')


## training parameters
parser.add_argument('-nepochs', '--num-epochs', default=25, type=int, help='number of epochs to train.')
parser.add_argument('-pat', '--patience', default=15, type=int, help='number of epochs to wait before early stopping.')
parser.add_argument('-upred', '--update-predictor', action='store_true', help='flag to enable backpropagation through to predictor model')
parser.add_argument('-opt', '--optimizer', default='adadelta', help='set the optimizer (adadelta|adam|adagrad|rmsprop|sgd)')
parser.add_argument('-lrate', '--learning-rate', default=1.0, type=float, help='learning rate')
parser.add_argument('-wdecay', '--weight-decay', default=0, type=float, help='weight decay for regularizing optimization (default: 0)')
parser.add_argument('-dout','--dropout', default=0.0, type=float, help='dropout probability to be applied in the inputs (default: 0.0')
parser.add_argument('-cnorm', '--clip-norm', default=None, type=float, help='clip value to clip gradients by L2 norm')
parser.add_argument('-logafter', '--log-interval', default=100, type=int, help='logging interval (in number of minibatches trained)')
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
torch.backends.cudnn.deterministic = True

# setting dataset paths
src_trainset_path = args.train_prefix + '.' + args.source_suffix
hyp_trainset_path = args.train_prefix + '.' + args.hypothesis_suffix
score_trainset_path = args.train_prefix + '.' + args.scores_suffix

### initialize model
pred_model = None
if args.predictor_model:
    logger.info("loading predictor model")
    pred_checkpoint = torch.load(args.predictor_model)
    #predcitor_optimizer = pred_checkpoint['optimizer']
    pred_args = pred_checkpoint['args']
    Predictor = model_utils.set_predictor_arch(pred_args.architecture)
    pred_model = Predictor(pred_args).cuda()
    pred_model.load_state_dict(pred_checkpoint['state_dict'])
    # if predictor model is available, set the source and target vocab size to this. override the parameter setting here.
    logger.info("setting vocab size from predictor model: (src:{},hyp:{})".format(pred_args.num_source_vocab, pred_args.num_target_vocab))
    args.num_source_vocab = pred_args.num_source_vocab
    args.num_target_vocab = pred_args.num_target_vocab


# loading vocabularies
logger.info("loading vocabularies")
src_vocab, trg_vocab = pred_checkpoint['vocab']
args.source_vocab_size = src_vocab.vocab_size
args.target_vocab_size = trg_vocab.vocab_size

# validating predictor
if args.predictor_model:
    logger.info('validating predictor on predictor validation set.')
    pred_src_validset_path = pred_args.valid_prefix + '.' + pred_args.source_suffix
    pred_trg_validset_path = pred_args.valid_prefix + '.' + pred_args.target_suffix
    pred_validset_reader = ParallelReader(source_dataset_path=pred_src_validset_path,
                                        target_dataset_path=pred_trg_validset_path,
                                        num_batches_in_cache=None,
                                        source_max_length=pred_args.max_source_length,
                                        target_max_length=pred_args.max_target_length,
                                        batchsize=pred_args.valid_batchsize,
                                        shuffle_batches=False)

    valid_loss  = predictor_trainer.run_validation(pred_model, pred_validset_reader, (src_vocab, trg_vocab), debug=False)
    logger.info("validation loss = %.4f" % valid_loss)


# loading training set
logger.info("loading training set")
trainset_reader = QualityDataReader(source_dataset_path=src_trainset_path,
                                    hypothesis_dataset_path=hyp_trainset_path,
                                    scores_path=score_trainset_path,
                                    num_batches_in_cache=None,
                                    source_max_length=args.max_source_length,
                                    hypothesis_max_length=args.max_hypothesis_length,
                                    batchsize=args.batchsize,
                                    shuffle_batches=True)

# loading validation/test sets
def load_evaluation_set(prefix, src_suffix, hyp_suffix, scores_suffix, eval_batchsize):
    """ helper function to load evaluation datasets including validation set. """
    src_test_path = prefix + '.' + src_suffix
    hyp_test_path = prefix + '.' + hyp_suffix
    score_test_path = prefix + '.' + scores_suffix
    evalset_reader = QualityDataReader(source_dataset_path=src_test_path,
                                        hypothesis_dataset_path=hyp_test_path,
                                        scores_path=score_test_path,
                                        num_batches_in_cache=None,
                                        source_max_length=None,
                                        hypothesis_max_length=None,
                                        batchsize=eval_batchsize,
                                        shuffle_batches=False)
    return evalset_reader

# loading validationset
logger.info("loading validation set")
validset_reader = load_evaluation_set(args.valid_prefix, args.source_suffix, args.hypothesis_suffix, args.scores_suffix, args.valid_batchsize)

# loading testset
testset_readers = []
if args.test_prefixes:
    logger.info('loading test set(s)')
    for test_prefix in args.test_prefixes:
        testset_reader = load_evaluation_set(test_prefix, args.source_suffix, args.hypothesis_suffix, args.scores_suffix, args.test_batchsize)
        testset_readers.append(testset_reader)

#setting input dimension
logger.info('initializing estimator model')
Estimator = model_utils.set_estimator_arch(args.architecture)
est_model = Estimator(args, pred_model=pred_model).cuda()
total_params = sum(p.numel() for p in est_model.parameters() if p.requires_grad==True)
logger.info("total number of trainable parameters of the model: {}".format(total_params))

# starting training
logger.info("starting training")
trainset_reader.reset()
vocab = (src_vocab,trg_vocab)
train(est_model, pred_model, args, trainset_reader, vocab, validset_reader, testset_readers)
