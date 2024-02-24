import tensorflow as tf


class ArcFace(tf.keras.layers.Layer):
  def __init__(self, n_classes=10, s=64.0, m=0.50, regularizer=None, **kwargs):
    self.n_classes = n_classes
    self.s = s
    self.m = m
    self.regularizer = tf.keras.regularizers.get(regularizer)
    super(ArcFace, self).__init__()

  def build(self, input_shape):
    n = self.n_classes
    self.kernels = self.add_weight(
      name="kernel",
      shape=(input_shape[0][1], n),
      initializer="glorot_uniform",
      trainable=True,
      regularizer=self.regularizer,
    )
    super(ArcFace, self).build(input_shape)

  def call(self, inputs, training=None):
    x, y = inputs
    x = tf.math.l2_normalize(x, axis=1)
    W = tf.nn.l2_normalize(self.kernels, axis=0)
    logits = tf.matmul(x, W, name="original_target_logits")

    if not training:
      return self.s * logits

    sin_a = tf.sqrt(1 - tf.square(logits))
    cos_m = tf.cos(self.m)
    sin_m = tf.sin(self.m)
    phi = logits * cos_m - sin_a * sin_m
    th = logits - cos_m
    one_hot = tf.one_hot(y, self.n_classes)
    marginal_target_logits = tf.where(tf.cast(th > 0, tf.bool), phi, logits)
    diff = tf.subtract(marginal_target_logits, logits)
    out = self.s * (logits + tf.multiply(one_hot, diff))
    return out

  def compute_output_shape(self, input_shape):
    return (input_shape[0][0], self.n_classes)
