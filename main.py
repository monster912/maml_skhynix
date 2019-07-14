from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import time
import os

from model import MAML
from data import Data
from accumulator import Accumulator
from layers import get_train_op

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--save_freq', type=int, default=1000)

parser.add_argument('--n_train_iters', type=int, default=2000)
parser.add_argument('--n_test_iters', type=int, default=1000)

parser.add_argument('--way', type=int, default=5)
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--query', type=int, default=15)

parser.add_argument('--inner_lr', type=float, default=0.1)
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument('--metabatch', type=int, default=10)

args = parser.parse_args()

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

savedir = './results/run' \
    if args.savedir is None else args.savedir
if not os.path.isdir(savedir):
  os.makedirs(savedir)

# data loader
data = Data(args)
model = MAML(args)
net = model.get_loss_multiple()

def train():
  optim = tf.train.AdamOptimizer(args.meta_lr)
  train_op = get_train_op(optim, net['cent'], clip=[-10., 10.])

  saver = tf.train.Saver(tf.trainable_variables())
  logfile = open(os.path.join(savedir, 'train.log'), 'w')

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())

  # meta-train
  train_logger = Accumulator('cent', 'acc')
  train_to_run = [train_op, net['cent'], net['acc']]

  # meta-validation (or meta-test)
  test_logger = Accumulator('cent', 'acc')
  test_to_run = [net['cent'], net['acc']]

  for i in range(args.n_train_iters+1):
    # feed_dict
    epi = model.episodes
    placeholders = [epi['xtr'], epi['ytr'], epi['xte'], epi['yte']]
    episode = data.generate_episode(args, training=True,
        n_episodes=args.metabatch)
    fdtr = dict(zip(placeholders, episode))

    train_logger.accum(sess.run(train_to_run, feed_dict=fdtr))

    if i % 5 == 0:
      line = 'Iter %d start, learning rate %f' % (i, args.meta_lr)
      print('\n' + line)
      logfile.write('\n' + line + '\n')
      train_logger.print_(header='train', episode=i*args.metabatch,
          logfile=logfile)
      train_logger.clear()

    if i % 50 == 0:
      # meta-validation
      for j in range(10):
        # feed_dict
        epi = model.episodes
        placeholders = [epi['xtr'], epi['ytr'], epi['xte'], epi['yte']]
        episode = data.generate_episode(args, training=False,
            n_episodes=args.metabatch)
        fdte= dict(zip(placeholders, episode))
        test_logger.accum(sess.run(test_to_run, feed_dict=fdte))

      test_logger.print_(header='test ', episode=i*args.metabatch,
          logfile=logfile)
      test_logger.clear()

    if i % args.save_freq == 0:
      saver.save(sess, os.path.join(savedir, 'model'))

  logfile.close()

def test():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  saver = tf.train.Saver(net['weights'])
  saver.restore(sess, os.path.join(savedir, 'model'))

  acc = []
  for j in range(args.n_test_iters//args.metabatch):
    epi = model.episodes
    placeholders = [epi['xtr'], epi['ytr'], epi['xte'], epi['yte']]
    episode = data.generate_episode(args, training=False,
        n_episodes=args.metabatch)
    fdte= dict(zip(placeholders, episode))
    acc.append(100*sess.run(net['acc'], feed_dict=fdte))

  print('mean accuracy : %f'%np.mean(acc))

if __name__=='__main__':
  if args.mode == 'train':
    train()
  elif args.mode == 'test':
    test()
  else:
    raise ValueError('Invalid mode %s' % args.mode)
