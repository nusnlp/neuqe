def set_predictor_arch(arch):
  """ function to select predictor architecture """
  if arch == 'rnn':
    from .predictors.rnn import Predictor
  elif arch == 'cnn':
    from .predictors.cnn import Predictor
  else:
    raise NotImplementedError
  return Predictor

def set_estimator_arch(arch):
  """ function to select estimator architecture """
  if arch == 'rnn':
    from .estimators.rnn_estimator import RNNEstimator as Estimator
  elif arch == 'cnn':
    from .estimators.cnn_estimator import CNNEstimator as Estimator
  else:
    raise NotImplementedError
  return Estimator
