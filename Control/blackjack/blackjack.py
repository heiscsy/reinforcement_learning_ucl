from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

kGammar = 1
kSteps = 500000

def init():
  hand = np.random.randint(low=12, high=22)
  usable = np.random.randint(low=0, high=2)
  dealer = np.random.randint(low=2, high=12)
  return hand, usable, dealer

def dealerPlay(dealer):
  while(dealer<17):
    dealer = dealer + np.random.randint(low=1, high=10)
  return dealer

def play(hand, dealer):
  terminate = False
  if hand in [20, 21]:
    terminate = True
    dealer_end = dealerPlay(dealer)
    if hand > dealer_end or dealer_end>21:
      reward = 1
    elif hand == dealer_end:
      reward = 0
    elif hand < dealer_end:
      reward = -1
  else:
    hand = hand + np.random.randint(low = 1, high = 10)
    reward = 0
    if hand>21:
      reward = -1
      terminate = True
  return terminate, hand, reward 

def updateValue(state, state_count, bp_trail, reward):
  gammar = 1
  for bp in reversed(bp_trail):
    state[bp[0], bp[1], bp[2]] = state[bp[0], bp[1], bp[2]] + gammar*reward
    state_count[bp[0], bp[1], bp[2]] = state_count[bp[0], bp[1], bp[2]] + 1
    gammar = gammar * kGammar
  return state, state_count

def Visualization(state, ax):
  X = np.arange(0, 10, 1)
  Y = np.arange(0, 10, 1)
  X, Y = np.meshgrid(X, Y)
  ax.plot_surface(X, Y, state[X, 0, Y])


def main():
  state = np.zeros((10, 2, 10))
  state_count = np.zeros((10, 2, 10))
  for steps in range(kSteps):
    terminate = False
    hand, usable, dealer = init()
    bp_trail = []
    print(terminate,',',hand)
    while not terminate:
      bp_trail.append([hand-12, usable, dealer-2])
      terminate, hand, reward = play(hand, usable)
      print(terminate,',',hand)
    state, state_count = updateValue(state, state_count, bp_trail, reward)
  state = np.nan_to_num(state / state_count)
  print(state)
  # fig = plt.figure()
  # ax = fig.add_subplot(121, projection='3d')
  # Visualization(state, ax)

if __name__ == '__main__':
  main()