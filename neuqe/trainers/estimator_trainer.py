import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from ..io import io_utils
from ..utils import metric_utils

def train_step(train_input, scores, est_model, est_optimizer, est_criterion, clip_norm=1.0, debug=False):
    """ One training step with an update over a batch

    Args:
        train_input : tuple of source, target/hyp, source_mask, hyp/target mask.
        scores: the target scores for training the estimator
        est_model: the estimator model
        est_optimizer: the estimator optimizer
        est_criterion: the loss function for training the estimator
        clip_norm: value to clip the gradients based on norm
        debug: flag to enable debugging

    Returns:
        est_loss_value: computed loss is returned
    """

    # set model to train mode and set optimizer gradients to zero to start training
    est_model.train()
    est_optimizer.zero_grad()

    # set loss function to zero
    loss = 0

    #extract source sentence tokens and hyp sentence tokens from input
    source = train_input[0]
    hyp = train_input[1]
    source_mask = train_input[2]
    hyp_mask = train_input[3]

    # convert source sentence to torch variable
    source_input = Variable(torch.LongTensor(source)).cuda()
    source_input_mask = Variable(torch.LongTensor(source_mask)).cuda()

    # convert hypothesis sentence to cuda variable to be used as reference
    hyp_input = Variable(torch.LongTensor(hyp)).cuda()
    hyp_input_mask = Variable(torch.LongTensor(hyp_mask)).cuda()

    # setting references
    scores_ref = Variable(torch.FloatTensor(scores)).cuda()

    # preparing model input
    model_input = (source_input, source_input_mask, hyp_input, hyp_input_mask)
    est_scores, log_probs = est_model(model_input)

    # computing the loss for estimator
    est_loss = est_criterion(est_scores, scores_ref)

    # computing gradients
    est_loss.backward()

    # clip by grad norm
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm(est_model.parameters(),max_norm=clip_norm)

    # updating parameters
    est_optimizer.step()

    # extracting the loss value
    est_loss_value = est_loss.data[0]

    # deleting Variables
    del est_loss, source_input, source_input_mask, hyp_input, hyp_input_mask, log_probs, est_scores, scores_ref

    return est_loss_value

def run_validation(est_model, validset_reader, vocab, est_criterion, metrics = [], debug=False):
    """ Function to perform validation

    Args:
        est_model: estimator model
        validset_reader: validation set data iterator
        vocab: vocabulary (source/target)
        est_criterion: the loss function to evaluate the estimator model
        metrics: the list of metrics used during validation
        debug: flag to be set to debug the function

    Returns:
        validation loss and scores for the evaluation metrics
    """
    est_model.eval()
    total_loss_value = 0
    minibatch_idx = 0
    validset_reader.reset()
    minibatch = validset_reader.next()
    all_est_scores = []
    all_ref_scores = []
    while(minibatch):
        minibatch_idx += 1
        pred_minibatch = [(src,hyp) for src,hyp,score in minibatch]
        scores = [score for src,hyp,score in minibatch]
        pred_valid_input = io_utils.create_predictor_input(minibatch,vocab)

        #extract source sentence tokens and target sentence tokens from input
        source = pred_valid_input[0]
        target = pred_valid_input[1]
        source_mask = pred_valid_input[2]
        target_mask = pred_valid_input[3]
        source_input = Variable(torch.LongTensor(source), volatile=True).cuda()
        source_input_mask = Variable(torch.LongTensor(source_mask), volatile=True).cuda()
        target_ref = Variable(torch.LongTensor(target), volatile=True).cuda()
        target_ref_mask = Variable(torch.LongTensor(target_mask), volatile=True).cuda()
        target_length = target_ref.size()[0]
        scores_ref = Variable(torch.FloatTensor(scores)).cuda()

        model_input = (source_input, source_input_mask, target_ref, target_ref_mask)
        est_scores, log_probs = est_model(model_input)

        est_loss = est_criterion(est_scores, scores_ref)

        all_est_scores.append(est_scores.data)
        all_ref_scores += scores

        total_loss_value += (est_loss.data[0])
        del est_loss, source_input, source_input_mask, target_ref, target_ref_mask, log_probs, est_scores, scores_ref
        minibatch = validset_reader.next()

    all_est_scores = list(torch.cat(all_est_scores,dim=0).cpu().numpy().reshape(-1))
    metric_scores = metric_utils.score(metrics, all_est_scores, all_ref_scores)

    avg_loss = total_loss_value / (minibatch_idx)
    return avg_loss, metric_scores


