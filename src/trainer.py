import os
import time
import datetime

import absl
import tensorflow as tf

from metrics import Metrics

logger = absl.logging


class Trainer(object):

  def __init__(
    self, model, logdir, optimizer, loss_fn, num_classes, input_shape,
    ckpt_steps=100, factor=0.5, patience=5, min_lr=1e-6):

    # model
    self.model = model
    self.logdir = logdir
    self.logging_steps = 1000
    self.num_classes = num_classes
    self.input_shape = input_shape
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.best_threshold = 0.97
    self.ckpt_steps = ckpt_steps
    self.train_log_steps = 20

    self.callbacks = self.get_callbacks(logdir)
    for callback in self.callbacks:
      callback.set_model(self.model)

    # metrics
    self.metrics = Metrics(num_classes)

    # summary
    self.summary_writer = tf.summary.create_file_writer(logdir)

    try:
      init_lr = optimizer.lr.numpy()
      self.reduce_lr = ReduceLearningRate(
        init_lr, factor=factor, patience=patience, min_lr=min_lr)
    except Exception as e:
      print(e)

  def get_callbacks(self, logdir, profile_batch=0):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=logdir, histogram_freq=1, profile_batch=profile_batch)
    return [tensorboard_callback]

  def train(self, dataset, logging_steps, lr=None):
    self.logging_steps = logging_steps

    optimizer = self.optimizer
    loss_fn = self.loss_fn

    train_dataset = dataset.train_inputs()
    valid_dataset = dataset.valid_inputs()

    with self.summary_writer.as_default():
      # Train loop
      step = self.train_loop(
        optimizer, loss_fn, train_dataset, valid_dataset)

      # Validation for final
      self.valid_loop(loss_fn, valid_dataset, step, True)

  def train_loop(
    self, optimizer, loss_fn, train_dataset, valid_dataset):

    def logging_train(y_true, y_pred, step):
      # train logging
      self.metrics.set_train_accuracy(y_true, y_pred)

      if step % self.train_log_steps == 0:

        # https://github.com/tensorflow/models/issues/7687
        lr = optimizer._decayed_lr(var_dtype=tf.float32)
        tf.summary.scalar('Learning Rate', data=lr, step=step)

        self.metrics.write_train(step)
        self.metrics.reset_train()

    # restore from checkpoints
    ckpt, manager, step = self.restore_checkpoints(optimizer)

    # elapsed time per steps
    global_step_time = time.time()

    for (x_train, y_train) in train_dataset:
      # manually update steps
      step += 1
      ckpt.step.assign(step)

      # training step
      y_true, y_pred = self.train_step(
        self.model, optimizer, loss_fn, x_train, y_train)

      logging_train(y_train, y_pred, step)

      if step % self.ckpt_steps == 0:
        self.save_checkpoints(ckpt, manager, step)
        elapsed_time = time.time() - global_step_time
        step_time = elapsed_time / self.ckpt_steps
        tf.summary.scalar('Global Step', data=step_time, step=step)
        logger.info(f'Global step: {step_time:.4f} sec')
        global_step_time = time.time()

      if step % self.logging_steps == 0:
        # save checkpoints
        start_time = time.time()
        self.save_checkpoints(ckpt, manager, step)
        logger.info(f'Save checkpoint: {time.time() - start_time:.4f} sec')

        start_time = time.time()
        self.save_train_images(x_train.numpy(), y_train.numpy(), step)
        logger.info(f'Save train images: {time.time() - start_time:.4f} sec')

        # validation for each steps
        start_time = time.time()
        self.valid_loop(loss_fn, valid_dataset, step)
        logger.info(f'Validation time: {time.time() - start_time:.4f} sec')
    return step

  def valid_loop(self, loss_fn, valid_dataset, step, output_image=False):
    def logging_valid(y_true, y_pred, embeddings):
      self.metrics.set_valid_accuracy(y_true, y_pred, embeddings)

    start_time = time.time()

    # Validation
    for (x_valid, y_valid) in valid_dataset:
      y_true, y_pred, prelogits = self.valid_step(
        self.model, loss_fn, x_valid, y_valid)
      embeddings = tf.math.l2_normalize(
        prelogits, axis=1, epsilon=1e-10, name='embeddings')
      logging_valid(y_true, y_pred, embeddings)

    start_time = time.time()
    self.metrics.write_valid(step)
    score = self.metrics.get_valid_accuracy()
    self.save_best_score_model(score)
    logger.info(f'Save metrics: {time.time() - start_time:.4f} sec')
    logger.info(f'Valid accuracy: {score:.4f}')
    self.metrics.reset_valid()

  def save(self, dirname=None):
    if dirname is None:
      dirname = datetime.datetime.now().strftime('%s')
    path = os.path.join(self.logdir, 'models', dirname)
    os.makedirs(path, exist_ok=True)
    self.convert(self.model, path)

  def convert(self, model, output_dir):
    Lambda = tf.keras.layers.Lambda

    # Image Preprocessing
    # image_size = self.input_shape[0]
    # input_shape = (None, None, 3)
    inputs = tf.keras.Input(self.input_shape, dtype=tf.uint8, name='inputs')
    # inputs = tf.image.resize_with_pad(inputs, image_size, image_size)
    x = tf.cast(inputs, dtype=tf.float32)
    x = tf.math.divide(x, 255.)

    # Get base model layer
    prelogits = model.layers[1](x)
    prelogits = Lambda(lambda x: x, name='prelogits')(prelogits)

    def l2_norm(x):
      return tf.math.l2_normalize(x, axis=1, epsilon=1e-10)
    embeddings = Lambda(l2_norm, name='embeddings')(prelogits)

    # Output type
    predictions = {
      'embeddings': embeddings, 'prelogits': prelogits
    }

    model = tf.keras.Model(inputs, predictions)
    print(model.summary())
    tf.saved_model.save(model, output_dir)

  def save_best_score_model(self, score):
    if score < self.best_threshold:
      return
    dirname = datetime.datetime.now().strftime('%s')
    dirname = '{}-{:.4f}'.format(dirname, score)
    self.save(dirname)
    self.best_threshold = score

  def restore_checkpoints(self, optimizer):
    step = 0
    ckpt_path = os.path.join(self.logdir, 'ckpts')
    ckpt = tf.train.Checkpoint(
      step=tf.Variable(step, tf.int64), optimizer=optimizer, net=self.model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
      step = ckpt.step.numpy()
      logger.info(
        'Restore from ckpt: {}, step={}'.format(
          manager.latest_checkpoint, step))
    else:
      logger.info('Initialize from scratch')
    return ckpt, manager, step

  def save_checkpoints(self, ckpt, manager, step):
    ckpt.step.assign(step)
    save_path = manager.save()
    logger.info('Save checkpoint for step {}: {}'.format(
      int(ckpt.step), save_path))

  @tf.function
  def train_step(
    self, model, optimizer, loss_fn, x_train, y_train):
    with tf.GradientTape() as tape:
      y_pred, prelogits = model([x_train, y_train], training=True)
      losses = loss_fn(y_train, y_pred)
    grads = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    self.metrics.set_train_loss([losses])
    return y_train, y_pred

  @tf.function
  def valid_step(self, model, loss_fn, x_valid, y_valid):
    y_pred, prelogits = model([x_valid, y_valid])
    losses = loss_fn(y_valid, y_pred)
    self.metrics.set_valid_loss([losses])
    return y_valid, y_pred, prelogits
