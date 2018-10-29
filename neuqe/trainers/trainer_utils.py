import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import logging
from ..utils import log_utils as L
logger = logging.getLogger(__name__)

def log_train_info(epoch_idx, minibatch_idx, total_loss_value, num_batches=None):
    """ Function to print log output during training """
    print_loss_avg = total_loss_value / minibatch_idx
    if num_batches:
        batch_info="%d/%d" %(minibatch_idx,num_batches)
    else:
        batch_info="%d" %(minibatch_idx)
    logger.info('epoch: %d \t| minibatch: %s \t| avg.loss=%.4f' % (epoch_idx, batch_info, print_loss_avg))

def save_checkpoint(state, save_after_epochs, is_best, no_save_best=False, model_path='model.pt', best_model_path='model_best.pt'):
    """ saving model during training """
    if save_after_epochs == True:
        logger.info('saving model at ' + model_path)
        torch.save(state, model_path)
    if is_best == True:
        logger.info('saving model at ' + best_model_path)
        if no_save_best == False:
            torch.save(state, best_model_path)

def set_criterion(loss):
    """ Setting criterion for training the estimator """
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'mae':
        criterion = nn.L1Loss()
    else:
        raise "NotImplementedError"
    return criterion

def set_optimizer(optimizer):
    """ Setting optimizer algorithm """
    if optimizer == 'adam':
        return optim.Adam
    elif optimizer == 'adadelta':
        return optim.Adadelta
    elif optimizer == 'adagrad':
        return optim.Adagrad
    elif optimizer == 'rmsprop':
        return optim.RMSprop
    elif optimizer == 'sgd':
        return optim.SGD
    else:
        raise NotImplementedError
