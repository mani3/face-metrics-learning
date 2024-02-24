import os
import glob
import math

import absl
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.ops import control_flow_ops

logger = absl.logging


class VGGFace2(object):
  def __init__(self, image_dir, image_size, epochs, batch_size, seed=1234, split_ratio=0.0, num_gpus=1):
    self.image_size = image_size
    self.epochs = epochs
    self.batch_size = batch_size

    self.image_dir = os.path.join(image_dir, "**", "*.jpg")
    self.image_paths = glob.glob(self.image_dir)
    self.label_ids = self.get_label_ids(self.image_paths)
    self.seed = tf.constant(seed, dtype=tf.int64) if isinstance(seed, int) else 999

    self.train_count = int(self.total_count() * split_ratio) if split_ratio > 0.0 else -1
    self.max_steps = int(epochs * (self.train_count / self.batch_size) * num_gpus)
    valid_count = self.total_count() - self.train_count
    logger.info(f"total_count: {self.total_count()}")
    logger.info(f"train_count: {self.train_count}, valid_count: {valid_count}")
    logger.info(f"max_steps: {self.max_steps}, steps_par_epoch: {self.steps_par_epoch()}")

  def total_count(self):
    return len(self.image_paths)

  def steps_par_epoch(self):
    return self.total_count() // self.batch_size

  def label_count(self):
    return len(self.label_ids)

  def label_index(self, label):
    label = label.numpy()
    return self.label_ids.index(str(label, "utf8"))

  def get_label_ids(self, image_paths):
    def split_path(path):
      return path.split("/")[-2]

    labels = [split_path(p) for p in image_paths]
    labels = list(sorted(set(labels)))
    return labels

  def train_inputs(self):
    batch_size = self.batch_size
    epochs = self.epochs

    num_parallel = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.list_files(self.image_dir, seed=self.seed)
    dataset = (
      dataset.take(self.train_count)
      .shuffle(10000)
      .repeat(epochs)
      .map(self.load_image, num_parallel)
      .map(self.preprocessing_train, num_parallel)
      .batch(batch_size)
      .prefetch(num_parallel)
    )
    return dataset

  def valid_inputs(self):
    batch_size = self.batch_size
    num_parallel = tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.list_files(self.image_dir, seed=self.seed)
    dataset = dataset.skip(self.train_count)
    dataset = (
      dataset.repeat(1)
      .map(self.load_image, num_parallel)
      .map(self.preprocessing_valid, num_parallel)
      .batch(batch_size)
      .prefetch(num_parallel)
    )
    return dataset

  def load_image(self, filename):
    name = tf.strings.split([filename], "/")
    label = name.values[-2]
    label = tf.py_function(self.label_index, [label], (tf.int64))
    image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
    image = tf.cast(image, tf.float32)
    return (image, label)

  def preprocessing_train(self, image, label):
    image_size = self.image_size
    image = random_color(image)
    image = random_flip(image)
    image = tf.image.resize_with_pad(image, image_size, image_size)
    image = random_erasing(image)
    image.set_shape([image_size, image_size, 3])
    image = random_rotation(image, 30.0)
    image = random_translation(image, 15.0)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

  def preprocessing_valid(self, image, label):
    image_size = self.image_size
    image = tf.image.resize_with_pad(image, image_size, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def random_erasing_np(image, p=0.5, s=(0.02, 0.4), r=(0.3, 1 / 0.3)):
  if np.random.rand() > p:
    return image
  mask_value = np.random.randint(0, 256)
  h, w, _ = image.shape
  mask_area = np.random.randint(h * w * s[0], h * w * s[1])
  mask_aspect_ratio = np.random.rand() * r[1] + r[0]
  mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
  mask_height = min(mask_height, h - 1)
  mask_width = int(mask_aspect_ratio * mask_height)
  mask_width = min(mask_width, w - 1)
  top = np.random.randint(0, h - mask_height)
  left = np.random.randint(0, w - mask_width)
  bottom = top + mask_height
  right = left + mask_width
  image[top:bottom, left:right, :].fill(mask_value)
  return image


def random_erasing(image, p=0.5, s=(0.02, 0.4), r=(0.3, 1 / 0.3)):
  def random_erasing_func(image):
    image = random_erasing_np(image.numpy(), p, s, r)
    return image

  image = tf.py_function(random_erasing_func, [image], (tf.float32))
  return image


def random_flip(image):
  image = tf.image.random_flip_up_down(image)
  image = tf.image.random_flip_left_right(image)
  return image


def random_rotation(image, k=0):
  angle = tf.random.uniform([], minval=-k, maxval=k, dtype=tf.float32)
  radian = angle * math.pi / 180.0
  return tfa.image.rotate(image, radian)


def random_translation(image, k=0):
  delta = tf.random.uniform([2], minval=-k, maxval=k, dtype=tf.float32)
  return tfa.image.translate(image, delta)


def apply_with_random_selector(x, func, num_cases):
  sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
  return control_flow_ops.merge(
    [func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case) for case in range(num_cases)]
  )[0]


def random_color(image):
  def random_color_fast(image, k=0):
    if k == 0:
      image = tf.image.random_saturation(image, 0.5, 1.5)
      image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    else:
      image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
      image = tf.image.random_saturation(image, 0.5, 1.5)
    return image

  image = apply_with_random_selector(image, lambda x, k: random_color_fast(x, k), num_cases=2)
  image = tf.clip_by_value(image, 0, 255)
  return image
