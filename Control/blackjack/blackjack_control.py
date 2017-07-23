from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

kGammar = 1
kEpsilon = 0.0
kSteps = 500000

def init():
  hand = np.random.randint(low=12, high=22)
  usable = np.random.randint(low=0, high=2)
  dealer = np.random.randint(low=2, high=12)
  return hand, usable, dealer

def dealerPlay(dealer):
  while(dealer<17):
    card = np.random.randint(low=1, high=10)
    if card ==1 and dealer +11 <= 21:
      dealer = dealer + 11
    else:
      dealer = dealer + card 
  return dealer

def play(hand, usable, dealer, action):
  terminate = False
  if action==1:
    terminate = True
    dealer_end = dealerPlay(dealer)
    if hand > dealer_end or dealer_end>21:
      reward = 1
    elif hand == dealer_end:
      reward = 0
    elif hand < dealer_end:
      reward = -1
  else:
    hand = hand + np.random.randint(low=1, high=10)
    reward = 0
    if hand>21:
      if usable == 1:
        hand = hand - 10
        usable = 0
      else:
        reward = -1
      terminate = True
  return terminate, hand, reward, usable 

def updateValue(state, state_count, bp_trail, reward):
  gammar = 1
  for bp in reversed(bp_trail):
    state_count_ = state_count[bp[0], bp[1], bp[2], bp[3]] + 1
    state_count[bp[0], bp[1], bp[2], bp[3]] = state_count_
    state[bp[0], bp[1], bp[2], bp[3]] =  1/state_count_*(gammar*reward - state[bp[0], bp[1], bp[2], bp[3]]) +\
       state[bp[0], bp[1], bp[2], bp[3]]
    gammar = gammar * kGammar
  return state, state_count

def updatePolicy(state_action, policy, bp_trail, epslon = kEpsilon):
  policy_new = policy
  #random_action = (np.random.random(np.shape(policy))<0.5).astype(np.int)
  #random_idx = np.random.random(np.shape(policy)) < epslon
  for bp in reversed(bp_trail):
    policy_new[bp[0], bp[1], bp[2]] = (state_action[bp[0],bp[1],bp[2],1]>state_action[bp[0],bp[1],bp[2],0]).astype(np.int)
  #policy_new[random_idx] = random_action[random_idx]
  return policy_new

def Visualization(state, ax):
  X = np.arange(0, 10, 1)
  Y = np.arange(0, 10, 1)
  X, Y = np.meshgrid(X, Y)
  ax.plot_surface(X, Y, state[X, 0, Y])


def main():
  state_action = np.zeros((10, 2, 10, 2))
  state_action_count = np.zeros((10, 2, 10, 2))
  policy = np.zeros((10, 2, 10)).astype(np.int)
  policy[[8,9], :, :]= 1
  print(policy)
  count = 0
  for _ in range(1000000):
    terminate = False
    hand, usable, dealer = init()
    initial = True
    bp_trail = []
    while not terminate:
      if initial:
        action = np.random.randint(low=0, high=2)
      else:
        action = policy[hand-12, usable, dealer-2]
      bp_trail.append([hand-12, usable, dealer-2, action])
      terminate, hand, reward, usable = play(hand, usable, dealer, action)
      initial = False
    count = count +1
    state_action, state_action_count = updateValue(state_action, state_action_count, bp_trail, reward)
    policy=updatePolicy(state_action, policy, bp_trail)
  print(state_action)
  print(policy[:,0,:])
  print(policy[:,1,:])

  # fig = plt.figure()
  # ax = fig.add_subplot(121, projection='3d')
  # Visualization(state, ax)

if __name__ == '__main__':
  main()