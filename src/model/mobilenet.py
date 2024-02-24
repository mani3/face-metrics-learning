import functools
import tensorflow as tf


def create_model(input_shape, num_classes, **kwargs):
  weights = kwargs.get("weights", "imagenet")
  alpha = kwargs.get("alpha", 1.0)

  model = tf.keras.applications.MobileNetV2(
    input_shape=input_shape, alpha=alpha, weights=weights, include_top=False, pooling="avg"
  )
  x = model.output
  x = tf.keras.layers.Dense(num_classes, activation=None, name="prelogits")(x)
  return tf.keras.models.Model(model.inputs, x)


class MobileNet(object):
  v2 = functools.partial(create_model, weights=None)
