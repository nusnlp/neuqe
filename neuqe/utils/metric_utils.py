import logging
from collections import OrderedDict
import math

logger = logging.getLogger(__name__)

def score(metrics, pred, ref):
    """ Function to score and print custom metrics """
    score_dict = OrderedDict()
    if metrics:
        for metric in metrics:
            if metric == 'pc':
                score_dict[metric] = pearson_correlation(pred,ref)
            elif metric == 'mae':
                score_dict[metric] = mean_absolute_error(pred, ref)
            elif metric == 'mse':
                score_dict[metric] = mean_squared_error(pred, ref)
            elif metric == 'rmse':
                score_dict[metric] = root_mean_squared_error(pred, ref)
            else:
                logger.error('Invalid metric: %s',metric)

    return score_dict

def pearson_correlation(pred, ref):
    """ Computes Pearson correlation """
    from scipy.stats import pearsonr
    pc = pearsonr(pred, ref)
    return pc[0]  # return correlation value and ignore p,value

def mean_absolute_error(pred, ref):
    """ Computes mean absolute error  """
    import sklearn.metrics
    mae = sklearn.metrics.regression.mean_absolute_error(pred, ref)
    return mae

def mean_squared_error(pred,ref):
    """ Computes mean squared error """
    import sklearn.metrics
    mse = sklearn.metrics.regression.mean_squared_error(pred, ref)
    return mse

def root_mean_squared_error(pred,ref):
    """ Computes root mean squared error """
    mse = mean_squared_error(pred,ref)
    rmse = math.sqrt(mse)
    return rmse