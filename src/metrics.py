import tensorflow as tf


class Metrics(object):

  def __init__(self, num_classes):
    self.num_classes = num_classes

    m = tf.keras.metrics
    loss_types = ['total']

    self.metrics_loss_train = []
    self.metrics_loss_valid = []

    for t in loss_types:
      name = 'Loss/train/{}'.format(t)
      l = m.Mean(name, dtype=tf.float32)
      self.metrics_loss_train.append(l)
      name = 'Loss/valid/{}'.format(t)
      l = m.Mean(name, dtype=tf.float32)
      self.metrics_loss_valid.append(l)

    self.accuracy_train = m.SparseCategoricalAccuracy('Accuracy/train')
    self.accuracy_valid = m.SparseCategoricalAccuracy('Accuracy/valid')
    self.auc_valid = m.AUC(name='AUC/valid')

    self.eval_cos_positive = m.Mean('Eval/consine/positive')
    self.eval_cos_negative = m.Mean('Eval/consine/negative')
    self.eval_embedding_mean = m.Mean('Eval/embedding/mean')

  def set_train_loss(self, logs: list):
    for m, l in zip(self.metrics_loss_train, logs):
      m(l)

  def set_train_accuracy(self, y_true, y_pred):
    self.accuracy_train(y_true, y_pred)

  def write_train(self, step):
    for m in self.metrics_loss_train:
      tf.summary.scalar(m.name, m.result(), step=step)
    m = self.accuracy_train
    tf.summary.scalar(m.name, m.result(), step=step)

  def reset_train(self):
    for m in self.metrics_loss_train:
      m.reset_states()
    self.accuracy_train.reset_states()

  def set_valid_loss(self, logs: list):
    for m, l in zip(self.metrics_loss_valid, logs):
      m(l)

  def set_valid_accuracy(self, y_true, y_pred, embeddings):
    self.accuracy_valid(y_true, y_pred)
    self.auc_valid(tf.one_hot(y_true, self.num_classes), y_pred)

    batch_size = y_true.shape[0]
    cos_theta = tf.matmul(embeddings, tf.transpose(embeddings))
    one_hot = tf.one_hot(y_true, self.num_classes)
    pair_labels = tf.matmul(one_hot, tf.transpose(one_hot))

    exclude_diag_mask = tf.abs(tf.linalg.diag(tf.ones(batch_size)) - 1)
    pair_cosine = tf.boolean_mask(
      cos_theta, tf.cast(exclude_diag_mask > 0, tf.bool))
    pair_filter = tf.boolean_mask(
      pair_labels, tf.cast(exclude_diag_mask > 0, tf.bool))
    pair_filter.set_shape([None])

    positive_pair = tf.boolean_mask(
      pair_cosine, tf.cast(pair_filter > 0, tf.bool))
    negative_pair = tf.boolean_mask(
      pair_cosine, tf.cast(pair_filter < 1, tf.bool))
    self.eval_cos_positive(positive_pair)
    self.eval_cos_negative(negative_pair)
    self.eval_embedding_mean(embeddings)

  def get_valid_accuracy(self):
    return self.accuracy_valid.result()

  def write_valid(self, step):
    for m in self.metrics_loss_valid:
      tf.summary.scalar(m.name, m.result(), step=step)
    metrics_list = [
      self.accuracy_valid,
      self.auc_valid,
      self.eval_cos_positive,
      self.eval_cos_negative,
      self.eval_embedding_mean
    ]
    for m in metrics_list:
      tf.summary.scalar(m.name, m.result(), step=step)

  def reset_valid(self):
    for m in self.metrics_loss_valid:
      m.reset_states()
    metrics_list = [
      self.accuracy_valid,
      self.auc_valid,
      self.eval_cos_positive,
      self.eval_cos_negative,
      self.eval_embedding_mean
    ]
    for m in metrics_list:
      m.reset_states()
