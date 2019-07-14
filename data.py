import numpy as np

class Data:
  def __init__(self, args):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('./data', one_hot=True, validation_size=0)
    x, y = mnist.train.images, mnist.train.labels
    y_ = np.argmax(y, axis=1)

    self.N = 1000 # total num instances per class
    self.Ktr = 5 # total num train classes
    self.Kte = 5 # total num test classes

    self.xtr = [x[y_==k][:self.N,:] for k in range(self.Ktr)]
    self.xte = [x[y_==k][:self.N,:] for k in range(self.Ktr, self.Ktr + self.Kte)]

  def generate_episode(self, args, training=True, n_episodes=1):
    generate_label = lambda way, n_samp: np.repeat(np.eye(way), n_samp, axis=0)
    n_way, n_shot, n_query = args.way, args.shot, args.query
    K = self.Ktr if training else self.Kte
    x = self.xtr if training else self.xte

    xs, ys, xq, yq = [], [], [], []
    for t in range(n_episodes):
      # sample WAY classes
      classes = np.random.choice(range(K), size=n_way, replace=False)

      support_set = []
      query_set = []
      for k in list(classes):
        # sample SHOT and QUERY instances
        idx = np.random.choice(range(self.N), size=n_shot+n_query, replace=False)
        x_k = x[k][idx]
        support_set.append(x_k[:n_shot])
        query_set.append(x_k[n_shot:])

      xs_k = np.concatenate(support_set, 0)
      xq_k = np.concatenate(query_set, 0)
      ys_k = generate_label(n_way, n_shot)
      yq_k = generate_label(n_way, n_query)

      xs.append(xs_k)
      xq.append(xq_k)
      ys.append(ys_k)
      yq.append(yq_k)

    xs, ys = np.stack(xs, 0), np.stack(ys, 0)
    xq, yq = np.stack(xq, 0), np.stack(yq, 0)
    return [xs, ys, xq, yq]
