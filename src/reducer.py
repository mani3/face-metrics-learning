import numpy as np


class ReduceLearningRate(object):
  def __init__(self, init_lr, factor=np.sqrt(0.1), patience=5, mode="max", min_delta=1e-4, min_lr=0):
    self.current_lr = init_lr
    self.factor = factor
    self.patience = patience
    self.mode = mode
    self.min_delta = min_delta
    self.min_lr = min_lr
    self.wait = 0
    self.best = 0
    self.mode = mode
    self.monitor_op = None
    self._reset()

  def _reset(self):
    if self.mode == "min":
      self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
      self.best = np.Inf
    else:
      self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
      self.best = -np.Inf
    self.wait = 0

  def new_lr(self, epoch, metrics_value):
    if self.monitor_op(metrics_value, self.best):
      self.best = metrics_value
      self.wait = 0
    else:
      self.wait += 1
      if self.wait >= self.patience:
        old_lr = self.current_lr
        if old_lr > self.min_lr:
          new_lr = old_lr * self.factor
          new_lr = max(new_lr, self.min_lr)
          self.current_lr = new_lr
          self.wait = 0
          print("Epoch %05d: ReduceLearningRate reducing learning " "rate to %s." % (epoch + 1, new_lr))
    return self.current_lr
