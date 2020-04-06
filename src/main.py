import os
import datetime

import absl
from absl import app
from absl import flags

import tensorflow as tf

import loss
from model.mobilenet import MobileNet
from layer import arcface

from dataset.dataset import VGGFace2
from trainer import Trainer


logger = absl.logging


FLAGS = flags.FLAGS
flags.DEFINE_float('validation_split_ratio', 0.2, 'Validation data split ratio')

flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate')
flags.DEFINE_integer('learning_rate_decay_steps', 0, 'Learning rate decay steps')
flags.DEFINE_float('learning_rate_decay_rate', 1.0, 'Learning rate decay rate')

flags.DEFINE_float('mobilenet_alpha', 1.0, 'MobileNet alpha')

flags.DEFINE_integer('image_size', 128, 'Image size')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('epochs', 20, 'Epochs')
flags.DEFINE_integer('embedding_size', 512, 'Embedding size')
flags.DEFINE_integer('seed', None, 'Random seed')

flags.DEFINE_string('model_name', 'mobilenetv2', 'Model networks')
flags.DEFINE_string('loss_method', 'arcface', 'Loss type: arcface, am_softmax, adacos')
flags.DEFINE_string('loss_name', 'softmax_loss', 'Loss functions(focal_loss or multi_loss)')
flags.DEFINE_string('optimizer_name', 'adam', 'Optimizer name("adam", "momentum", "rmsprop")')

flags.DEFINE_string('input_dir', './train_faces', 'Training input dir')
flags.DEFINE_string('output_dir', './outputs', 'Outputs dir')
flags.DEFINE_string('h5_model_path', None, 'Pretrained saved model path')

flags.DEFINE_integer('num_gpus', 1, 'Number of gpu')


def get_optimizer(name):
  lr = FLAGS.learning_rate

  if name == 'rmsprop':
    return tf.keras.optimizers.RMSprop(learning_rate=lr)
  elif name == 'adam':
    return tf.keras.optimizers.Adam(learning_rate=lr)
  elif name == 'sgd':
    return tf.keras.optimizers.SGD(
      learning_rate=lr, momentum=0.9, nesterov=True)
  else:
    raise ValueError('Unknown Optimizer:', name)


def get_model(
  base_model, loss_method, input_shape, num_classes,
  embedding_size, **kwargs):
  # Image inputs
  inputs = tf.keras.layers.Input(input_shape)

  model = MobileNet.v2(input_shape, embedding_size, **kwargs)

  logger.info('Base network: ')
  logger.info(model.summary())

  # Embedding model
  model = tf.keras.Model(inputs, model(inputs))

  prelogits = model.output  # 512 dims
  y = tf.keras.layers.Input(shape=[num_classes, ], dtype=tf.int32)

  if loss_method == 'arcface':
    x = arcface.ArcFace(num_classes, **kwargs)([prelogits, y])
  else:
    raise ValueError('Not found model name:', loss_method)
  x = tf.keras.layers.Softmax()(x)
  model = tf.keras.Model([model.inputs, y], [x, prelogits])
  return model


def get_loss(name):
  return loss.softmax_loss()


def run(logdir, model_name, opt_name, loss_name):
  if FLAGS.seed is not None:
    tf.random.set_seed(FLAGS.seed)

  epochs = FLAGS.epochs
  input_dir = FLAGS.input_dir
  image_size = FLAGS.image_size
  batch_size = FLAGS.batch_size
  input_shape = [image_size, image_size, 3]
  valid_split_ratio = FLAGS.validation_split_ratio

  dataset = VGGFace2(
    input_dir, image_size, epochs, batch_size,
    split_ratio=valid_split_ratio,
    num_gpus=FLAGS.num_gpus)
  num_classes = dataset.label_count()
  steps_par_epoch = dataset.steps_par_epoch()

  logger.info(f'model_name: {model_name}')
  logger.info(f'opt_name: {opt_name}')

  model = get_model(model_name, FLAGS.loss_method,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    embedding_size=FLAGS.embedding_size,
                    alpha=FLAGS.mobilenet_alpha)
  logger.info(model.summary())

  if FLAGS.h5_model_path:
    model.load_weights(FLAGS.h5_model_path)
    logger.info(f'Loaded weights: {FLAGS.h5_model_path}')

  optimizer = get_optimizer(opt_name)
  loss = get_loss(loss_name)

  trainer = Trainer(model, logdir, optimizer, loss, num_classes, input_shape)
  with trainer.summary_writer.as_default():
    tf.summary.text('parameters', FLAGS.flags_into_string(), step=0)

  logging_steps = steps_par_epoch
  logger.info(f'logging_steps: {logging_steps}')

  trainer.train(dataset, logging_steps, FLAGS.learning_rate)
  trainer.save()


def main(args):
  logger.info(FLAGS.flags_into_string())
  if FLAGS.output_dir:
    logdir = FLAGS.output_dir
  else:
    logdir = os.path.join(
      './logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
  run(logdir, FLAGS.model_name, FLAGS.optimizer_name, FLAGS.loss_name)


if __name__ == "__main__":
  app.run(main)
