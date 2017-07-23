from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from world import GridWorld


kStateNum = 4*12
kActionNum = 4
kEpisodes = 500
kEpsilon = 0.1
kAlpha = 0.5


def pos_to_state(pos):
  return pos[0]*12 + pos[1]

def state_to_pos(state):
  return [state/12, state%12]

def epsilonGreedy(state, Q):
  if np.random.rand()<1-kEpsilon:
    action = np.argmax(Q[state,:])
  else:
    action = np.random.randint(low=0, high=4)
  return action

def main():
  Q = np.zeros([kStateNum, 4])
  world = GridWorld()
  for eps in range(kEpisodes):
    reward_acc = 0
    state = pos_to_state(world.init())
    while True:
      action = epsilonGreedy(state, Q)
      pos_, reward, terminate = world.play(action)
      reward_acc = reward_acc+reward
      state_ = pos_to_state(pos_)
      Q[state, action] = Q[state, action] + \
                        kAlpha * (reward + \
                        np.max(Q[state_, :]) - \
                        Q[state, action])
      state = state_
      if terminate:
        print("EPS: %d - Reward: %d"%(eps, reward_acc))
        break


if __name__ == '__main__':
  main()