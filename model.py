from layers import *

class MAML:
  def __init__(self, args):
    self.xdim = 28
    self.input_channel, self.n_channel = 1, 32
    self.way = args.way # num of classes per each episode
    self.inner_lr = args.inner_lr # inner gradient stepsize
    self.metabatch = args.metabatch # metabatch size

    xshape = [self.metabatch, None, self.xdim*self.xdim*self.input_channel]
    yshape = [self.metabatch, None, self.way]
    self.episodes = {
        'xtr': tf.placeholder(tf.float32, xshape, name='xtr'),
        'ytr': tf.placeholder(tf.float32, yshape, name='ytr'),
        'xte': tf.placeholder(tf.float32, xshape, name='xte'),
        'yte': tf.placeholder(tf.float32, yshape, name='yte')}

    self.conv_init = tf.truncated_normal_initializer(stddev=0.02)
    self.fc_init = tf.random_normal_initializer(stddev=0.02)
    self.bias_init = tf.zeros_initializer()

  def get_theta(self, reuse=None):
    with tf.variable_scope('theta', reuse=reuse):
      theta = {}
      for l in [1,2,3,4]:
        indim = self.input_channel if l == 1 else self.n_channel
        theta['conv%d_w'%l] = tf.get_variable('conv%d_w'%l,
            [3, 3, indim, self.n_channel], initializer=self.conv_init)
        theta['conv%d_b'%l] = tf.get_variable('conb%d_b'%l,
            [self.n_channel], initializer=self.bias_init)
      theta['dense_w'] = tf.get_variable('dense_w',
          [self.n_channel, self.way], initializer=self.fc_init)
      theta['dense_b'] = tf.get_variable('dense_b',
          [self.way], initializer=self.bias_init)
      return theta

  '''
  def get_alpha(self, reuse=None):
    with tf.variable_scope('alpha', reuse=reuse):
      alpha = {}
      for l in [1,2,3,4]:
        indim = self.input_channel if l == 1 else self.n_channel
        alpha['conv%d_w'%l] = tf.get_variable('conv%d_w'%l,
            [3, 3, indim, self.n_channel], initializer=self.conv_init)
        alpha['conv%d_b'%l] = tf.get_variable('conb%d_b'%l,
            [self.n_channel], initializer=self.bias_init)
      alpha['dense_w'] = tf.get_variable('dense_w',
          [self.n_channel, self.way], initializer=self.fc_init)
      alpha['dense_b'] = tf.get_variable('dense_b',
          [self.way], initializer=self.bias_init)
      return alpha
  '''

  def forward(self, x, theta):
    x = tf.reshape(x, [-1, self.xdim, self.xdim, self.input_channel])
    for l in [1,2,3,4]:
      w, b = theta['conv%d_w'%l], theta['conv%d_b'%l]
      x = conv_block(x, w, b, bn_scope='conv%d_bn'%l)
    return tf.matmul(flatten(x), theta['dense_w']) + theta['dense_b']

  def get_loss_single(self, inputs):
    xtr, ytr, xte, yte = inputs
    theta = self.get_theta(reuse=False)

    train_logits = self.forward(xtr, theta)
    train_loss = cross_entropy(train_logits, ytr)

    grads = tf.gradients(train_loss, theta.values()) # compute gradients
    gradients = dict(zip(theta.keys(), grads))

    theta = dict(zip(theta.keys(),
      [theta[key] - self.inner_lr * gradients[key] for key in theta.keys()]))

    test_logits = self.forward(xte, theta)
    cent = cross_entropy(test_logits, yte)
    acc = accuracy(test_logits, yte)
    return cent, acc

  def get_loss_multiple(self, reuse=None):
    xtr, ytr = self.episodes['xtr'], self.episodes['ytr']
    xte, yte = self.episodes['xte'], self.episodes['yte']

    # map_fn: enables parallization
    cent, acc = tf.map_fn(
        self.get_loss_single,
        elems=(xtr, ytr, xte, yte),
        dtype=(tf.float32, tf.float32),
        parallel_iterations=self.metabatch)

    net = {}
    net['cent'] = tf.reduce_mean(cent)
    net['acc'] = tf.reduce_mean(acc)
    net['weights'] = tf.trainable_variables()
    return net
