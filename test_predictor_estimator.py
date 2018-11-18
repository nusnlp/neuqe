import os, sys
import argparse
import logging
import time
import shutil
from collections import defaultdict

import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import numpy as np

from neuqe.utils import log_utils as L
from neuqe.utils import metric_utils
from neuqe.trainers import predictor_trainer, estimator_trainer
from neuqe.trainers import trainer_utils
from neuqe.models import model_utils
from neuqe.io import io_utils
from neuqe.io.vocab_manager import VocabManager
from neuqe.io.qualitydata_reader import QualityDataReader
logger = logging.getLogger(__name__)


def test(est_model, est_args, args, test_samples, vocab, test_scores=None):
    est_model.eval()
    sample_idx = 0
    loss = 0
    total_loss_value = 0
    est_criterion = trainer_utils.set_criterion(est_args.loss)
    out_scores = []
    for sample in test_samples:
        sample_as_batch = [sample]
        pred_input = io_utils.create_predictor_input(sample_as_batch, vocab)

        #extract source sentence tokens and target sentence tokens from input
        source = pred_input[0]
        target = pred_input[1]
        source_mask = pred_input[2]
        target_mask = pred_input[3]

        # convert to autograd Variables
        source_input = Variable(torch.LongTensor(source), volatile=True).cuda()
        source_mask_input = Variable(torch.LongTensor(source_mask), volatile=True).cuda()
        target_ref = Variable(torch.LongTensor(target), volatile=True).cuda()
        target_ref_mask = Variable(torch.LongTensor(target_mask), volatile=True).cuda()
        target_length = target_ref.size()[0]

        model_input = (source_input, source_mask_input, target_ref, target_ref_mask)
        est_score, log_probs= est_model(model_input)

        out_scores.append(est_score.data[0][0]) # only one element in output
        if test_scores:
            scores_ref = Variable(torch.FloatTensor([test_scores[sample_idx]])).cuda()
            est_loss = est_criterion(est_score, scores_ref)
            total_loss_value +=     (est_loss.data[0])

        sample_idx += 1
        if(args.debug==True):
            return 0.0,0.0
    assert sample_idx == len(test_samples), "error in dimension of samples and testset"
    if test_scores:
        avg_loss = total_loss_value / len(test_samples)
    else:
        avg_loss = None
    return out_scores, avg_loss

##############
## Arguments
###############

parser = argparse.ArgumentParser()
parser.add_argument('-test', '--test-prefix', nargs='+', required=True,  help='prefix of path to test dataset (without extension suffix).')
parser.add_argument('-ssuf', '--source-suffix',  required=True, help='extension suffix of source part.')
parser.add_argument('-hsuf', '--hypothesis-suffix', required=True,  help='extension suffix of hypothesis part.')
parser.add_argument('-scoresuf', '--scores-suffix', required=True, help='extension suffix of ref/output scores file.')
parser.add_argument('-emodel','--estimator-model', dest='pemodels', nargs='*', action='append', required=True,  help='path to trained estimator model (format, <optional-pred model path1 > ... <optional-pred-model pathN> est-model path) ')
parser.add_argument('-metrics', nargs='+', help='space separated metrics to evaluate on, e.g pc,mae,mse (requires scikit)')
parser.add_argument('-outdir', '--output-dir', required=True, help='path to output directory')
parser.add_argument('-gm', '--geometric-mean', action='store_true', help='flag to average ensemble by geometric mean')
parser.add_argument('-debug', action='store_true', help='flag to enable debugging by training only single batch')
args = parser.parse_args()

# create output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# initialize logger
handlers = [logging.FileHandler(os.path.abspath(args.output_dir)+'/test_log.txt'), logging.StreamHandler()]
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG, datefmt='%d-%m-%Y %H:%M:%S', handlers = handlers)


##############
## Setting up
##############

test_ref_scores = defaultdict(list)
test_samples = defaultdict(list)

for test_prefix in args.test_prefix:
    # setting dataset paths
    src_testset_path = test_prefix + '.' + args.source_suffix
    hyp_testset_path = test_prefix + '.' + args.hypothesis_suffix

    # loadng test set
    logger.info("loading test set: {}".format(test_prefix))
    with open(src_testset_path,'r') as source_dataset, open(hyp_testset_path,'r') as hypothesis_dataset:
        for src_line, hyp_line in zip(source_dataset, hypothesis_dataset):
            src_tokens = src_line.strip().split()
            hyp_tokens = hyp_line.strip().split()
            test_samples[test_prefix].append((src_tokens, hyp_tokens))


    score_testset_path = test_prefix + '.' + args.scores_suffix
    if os.path.exists(score_testset_path):
        with open(score_testset_path,'r') as score_dataset:
            for score_line in score_dataset:
                ref_score = score_line.strip()
                test_ref_scores[test_prefix].append(float(ref_score))
    else:
        logger.warning("reference scores file is not found at " + score_testset_path + ". continuing prediction without evaluation.")


# loading estimator/predictor models
nmodels = len(args.pemodels)
final_out_scores = defaultdict(None)
for modelidx, pemodel in enumerate(args.pemodels):
    logger.info("************************************")
    logger.info("predictor/estimator model {}".format(modelidx))
    est_model_path = pemodel[-1]
    pred_model_path = None
    if len(pemodel) > 1:
        pred_model_path = pemodel[0]

    # loading the estimator model
    logger.info("loading estimator model: {}".format(est_model_path))
    est_checkpoint = torch.load(est_model_path)
    est_args = est_checkpoint['args']

    # loading the predictor model
    if not pred_model_path:
        if est_args.predictor_model:
            pred_model_path = est_args.predictor_model
        else:
            logger.warning("not using any predictor model")
            pred_model_path = None
    pred_model = None
    if pred_model_path:
        logger.info("loading predictor model: {}".format(pred_model_path))
        pred_checkpoint = torch.load(pred_model_path)
        pred_args = pred_checkpoint['args']

    #  initializing predictor model
    Predictor = model_utils.set_predictor_arch(pred_args.architecture)
    pred_model = Predictor(pred_args).cuda()
    pred_model.load_state_dict(pred_checkpoint['state_dict'])

    # setting predictor in evaluation mode
    for param in pred_model.parameters():
        param.requires_grad = False

    # initializing estimator model
    # setting architecture of estimator model
    Estimator = model_utils.set_estimator_arch(est_args.architecture)
    est_model = Estimator(est_args, pred_model=pred_model).cuda()
    # for backward compatibility
    est_model_state = est_model.state_dict()
    est_model_state.update(est_checkpoint['state_dict'])
    est_model.load_state_dict(est_model_state)

    # setting estimator in evaluation mode
    for param in est_model.parameters():
        param.requires_grad = False

    # check if est_args has num_source_vocab and num_target_vocab
    if not hasattr(est_args, 'num_source_vocab'):
        est_args.num_source_vocab = pred_args.num_source_vocab
    if not hasattr(est_args, 'num_target_vocab'):
        est_args.num_target_vocab = pred_args.num_target_vocab

    # loading vocabulary files
    logger.info("loading vocabularies")
    src_vocab, trg_vocab = pred_checkpoint['vocab']

    for test_prefix in args.test_prefix:
        logger.info("estimating scores of test set: {}".format(test_prefix))
        vocab = (src_vocab,trg_vocab)
        out_scores, test_loss = test(est_model, est_args, args, test_samples[test_prefix], vocab, test_ref_scores[test_prefix])

        if test_ref_scores[test_prefix] != []:
            logger.info('test set loss (%s) = %.4f' % (est_args.loss, test_loss) )
            if args.metrics:
                logger.info('evaluation on metrics for test set: {}'.format(test_prefix))
                metric_scores = metric_utils.score(metrics=args.metrics, pred=out_scores, ref=test_ref_scores[test_prefix])
                logger.info("{}={}".format(list(metric_scores.keys()),list(metric_scores.values())))

        if test_prefix in final_out_scores:
            if args.geometric_mean == True:
                final_out_scores[test_prefix] += np.log(np.array(out_scores))
            else:
                final_out_scores[test_prefix] += np.array(out_scores)
        else:
            if args.geometric_mean == True:
                final_out_scores[test_prefix] = np.log(np.array(out_scores))
            else:
                final_out_scores[test_prefix] = np.array(out_scores)

for test_prefix in args.test_prefix:
    final_out_scores[test_prefix] /= nmodels
    if args.geometric_mean == True:
        final_out_scores[test_prefix] = np.exp(final_out_scores[test_prefix])

    logger.info("************************************")
    if args.metrics:
        logger.info('final evaluation on testset: {}'.format(test_prefix))
        metric_scores = metric_utils.score(metrics=args.metrics, pred=list(final_out_scores[test_prefix]), ref=test_ref_scores[test_prefix])
        logger.info("{}={}".format(list(metric_scores.keys()),list(metric_scores.values())))


    output_scores_path = args.output_dir + '/' + os.path.basename(test_prefix) + '.' + args.scores_suffix + '.pred'
    logger.info("writing output to " + output_scores_path)
    with open(output_scores_path,'w') as fout_scores:
        for out_score in final_out_scores[test_prefix]:
            fout_scores.write("%.4f\n" %(out_score) )



