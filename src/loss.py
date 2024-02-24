import tensorflow as tf


def softmax_loss():
  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

  def loss(y_true, y_pred, tag="train"):
    losses = cross_entropy(y_true, y_pred)
    return losses

  return loss
