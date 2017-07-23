from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import gym
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

kEpisode = 9000
kEpsilon = 0.2
kAlpha = 0.01
kGammar = 1


class QNetwork:
  space_action = None
  q_label = None
  train_op = None
  q = None
  init = None
  loss = None
  g_v = None

  def __init__(self):
    self.space_action = tf.placeholder(dtype = tf.float32, shape=[None,6])
    self.q_label = tf.placeholder(dtype = tf.float32, shape=[None])
    self.create_net()
    self.init = tf.initialize_all_variables()


  def create_net(self):
    w1 = tf.get_variable("w1", shape = [6,10], 
      # initializer=tf.constant_initializer(0.0))
      initializer=tf.random_normal_initializer(stddev=1/60))
    b1 = tf.get_variable('b1', shape=[10], 
      initializer=tf.constant_initializer(0.0))
    # w2 = tf.get_variable("w2", shape=[10,10],
    #   initializer=tf.random_normal_initializer())
    w3 = tf.get_variable("w3", shape=[10,1],
      # initializer=tf.constant_initializer(0.0))
      initializer=tf.random_normal_initializer(stddev=1/10))
    b3 = tf.get_variable('b3', shape=[1], 
      initializer=tf.constant_initializer(0.0))
    l1 = tf.nn.relu(tf.matmul(self.space_action, w1)+b1)
    # l1 = tf.matmul(self.space_action, w1)+b1
    # l2 = tf.nn.relu(tf.matmul(l1, w3))
    self.q = tf.matmul(l1, w3)+b3
    # self.q = tf.matmul([self.space_action], w3)+b3
    weight_norm = tf.reduce_sum(0.00001*tf.convert_to_tensor([
        tf.nn.l2_loss(i) for i in [
        tf.get_collection('w1'), 
        tf.get_collection('w3')]]))
    self.loss = tf.reduce_mean(tf.square(self.q_label-self.q))+weight_norm
    
    opt= tf.train.AdamOptimizer(
      kAlpha)
    self.g_v = opt.compute_gradients(self.loss)
    self.train_op = opt.apply_gradients(self.g_v)

def buildFeature(space, action):
  feature = np.zeros(6)
  space_ = np.zeros(2)
  space_[0] = space[0]+0.5
  space_[1] = space[1]/0.07
  feature[action*2: action*2+2] = space_
  return feature

def Qvalue(space, action, qnet, session):
  feature = buildFeature(space, action)
  q_value = session.run([qnet.q], feed_dict={qnet.space_action: [feature]})
  return q_value

def epsilonGreedy(space, qnet, session):
  if np.random.uniform(0,1)< 1- kEpsilon:
    Qlist = []
    for a in range(0,3):
      feature = buildFeature(space, a)
      # print(feature)
      q_value = session.run([qnet.q], feed_dict={qnet.space_action: 
        [feature]})
      Qlist.append(q_value)
    action=np.random.choice(np.flatnonzero(
      np.array(Qlist) == np.array(Qlist).max()))
  else:
    action = np.random.randint(low=0, high=3)
  return action

# def updateW(space, action, delta, w):
#   feature = np.append(space, action)
#   # feature[1] = feature[1]/0.07
#   # feature[0] = feature[0]/1.2
#   return w + kAlpha*delta*feature

def main():
  # w = (np.random.random([5])-0.5)*20
  gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=1000,      # MountainCar-v0 uses 200
    # reward_threshold=-110.0,
  )
  env = gym.make('MountainCarMyEasyVersion-v0')
  # env = gym.make('MountainCar-v0')
  ExpBuffer = []
  schedule = 0
  qnet = QNetwork()
  with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.run([qnet.init])
    for eps in range(kEpisode):
      print('EPS %d:'%(eps))
      space = env.reset()
      count = 0
      while True:
        env.render()
        action = epsilonGreedy(space, qnet, sess)
        space_, reward, done, _ =env.step(action)
        ExpBuffer.append((space, action, reward, space_))
        schedule = schedule + 1
        if schedule==200:
          sample = np.random.permutation(range(200))
          sample_id = sample[0:64]
          feature = []
          q_label = [] 
          for id_ in sample_id:
            space, action, reward, space_ = ExpBuffer[id_]
            Qlist=[]
            for a in range(0, 3):
              Qlist.append(Qvalue(space_, a, qnet, sess))
            if space_[0]>=0.5:
              q_label.append(reward)
            else:
              q_label.append(reward+kGammar*np.max(np.array(Qlist)))
            
            feature.append(buildFeature(space, action))
          # print(feature)
          _, loss_, q_ = sess.run([qnet.train_op, qnet.loss, qnet.q], feed_dict={qnet.space_action:
              np.array(feature), qnet.q_label: np.array(q_label)})
          print(loss_)
          # print(q_)
          schedule = 0
        space = space_
        count = count + 1
        if done:
          print(count)
          break




if __name__=='__main__':
  main()