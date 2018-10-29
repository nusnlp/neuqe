import torch
import torch.nn as nn
from torch.autograd import Variable
from ..io import io_utils

def train_step(train_input, model, optimizer, criterion, clip_norm=1.0, debug=False):
    """ One training step with an update over a batch

    Args:
        train_input : tuple of source, target, source_mask, target mask.
        model: the predictor model
        optimizer: the optimizer algorithm
        criterion: the loss function for training
        clip_norm: value to clip the gradients based on norm
        debug: flag to enable debugging

    Returns:
        loss_value: computed loss is returned
    """

    # set model to train mode and set optimizer gradients to zero to start training
    model.train()
    optimizer.zero_grad()

    # set loss function to zero
    loss = 0

    # extract source sentence tokens and target sentence tokens from input
    # the target part of the train input also acts as the labels.
    source = train_input[0]
    target = train_input[1]
    source_mask = train_input[2]
    target_mask = train_input[3]

    # convert source sentence to torch variable
    source_input = Variable(torch.LongTensor(source), requires_grad=False).cuda()
    source_mask_input = Variable(torch.LongTensor(source_mask), requires_grad=False).cuda()
    if(debug==True): print("s/sm:",source_input,source_mask_input)
    if(debug==True): print("source_input_size:", source_input.size())

    # convert target sentence to cuda variable to be used as reference
    target_ref = Variable(torch.LongTensor(target), requires_grad=False).cuda()
    target_ref_mask = Variable(torch.LongTensor(target_mask), requires_grad=False).cuda()
    target_length = target_ref.size()[0]
    if(debug==True): print("t/tm:",target_ref,target_ref_mask)
    if(debug==True): print("target_reference_size:",target_ref.size())

    # preparing model input
    model_input = (source_input, source_mask_input, target_ref, target_ref_mask)

    # getting output
    log_probs, preqvecs, postqvecs = model(model_input)
    if(debug==True): print("target_var:", target_ref)

    # computing the loss
    loss = criterion(log_probs, target_ref[1:target_length-1].view(-1)) #[ti])
    if debug==True: print("loss:", loss)

    # computing gradients
    loss.backward()

    # grad clipping
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=clip_norm)

    # updating parameters
    optimizer.step()

    # extracting the loss value
    loss_value = loss.data[0]

    # deleting Variables
    del loss, source_input, source_mask_input, target_ref, target_ref_mask, log_probs, preqvecs, postqvecs

    return loss_value

def run_validation(model, validset_reader, vocab, debug=False):
    """ Function to perform validation

    Args:
        model: predictor model
        validset_reader: validation set data iterator
        vocab: vocabulary (source/target)
        debug: flag to be set to debug the function

    Returns:
        validation loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss_value = 0
    minibatch_idx = 0
    validset_reader.reset()
    minibatch = validset_reader.next()
    while(minibatch):
        minibatch_idx += 1
        valid_input = io_utils.create_predictor_input(minibatch,vocab)
        loss=0
        #extract source sentence tokens and target sentence tokens from input
        source = valid_input[0]
        target = valid_input[1]
        source_mask = valid_input[2]
        target_mask = valid_input[3]
        source_input = Variable(torch.LongTensor(source), requires_grad=False, volatile=True).cuda()
        source_mask_input = Variable(torch.LongTensor(source_mask), requires_grad=False, volatile=True).cuda()
        target_ref = Variable(torch.LongTensor(target), requires_grad=False, volatile=True).cuda()
        target_ref_mask = Variable(torch.LongTensor(target_mask), requires_grad=False, volatile=True).cuda()
        target_length = target_ref.size()[0]
        model_input = (source_input, source_mask_input, target_ref, target_ref_mask)
        log_probs,_,_ = model(model_input)
        loss = criterion(log_probs, target_ref[1:target_length-1].view(-1))
        total_loss_value += (loss.data[0])
        del loss, log_probs, source_input, source_mask_input, target_ref, target_ref_mask
        minibatch = validset_reader.next()

    avg_loss = total_loss_value / minibatch_idx
    return avg_loss
