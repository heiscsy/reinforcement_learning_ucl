from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from world import GridWorld

import numpy as np

kEpisodes = 200
kStateNum = 10 * 7
kEpsilon = 0.1
kAlpha = 0.5
klambda = 0.9

def pos_to_state(pos):
  return pos[1]*10 + pos[0]

def state_to_pos(state):
  return [state%10, state/10]

def epsilon_greedy(Q, state):
  action_value = Q[state,:]
  if np.random.rand()<1 - kEpsilon:
    action = np.argmax(action_value)
  else:
    action = np.random.randint(low=0, high=4)
  return action

def main():
  world = GridWorld()
  Q = np.zeros([kStateNum, 4])
  # Q[pos_to_state(world.goal), :] = 0
  time = 0
  for eps in range(kEpisodes):
    E = np.zeros([kStateNum, 4])
    print('eps:', eps)
    state = pos_to_state(world.init())
    action = epsilon_greedy(Q, state)
    while True:
      pos_, reward, terminate = world.play(action)
      if terminate:
        state_ = pos_to_state(pos_)
        delta = reward- Q[state, action]
        E[state, action] = E[state, action] + 1
        for s in range(kStateNum): 
          for a in range(4):
            Q[s, a] = Q[s, a] + kAlpha * delta * E[s, a]
            E[s, a] = klambda * E[s, a]
        break
      else:
        state_ = pos_to_state(pos_)
        action_ = epsilon_greedy(Q, state_)
        delta = reward+Q[state_, action_] - Q[state, action]
        E[state, action] = E[state, action] + 1
        for s in range(kStateNum): 
          for a in range(4):
            Q[s, a] = Q[s, a] + kAlpha * delta * E[s, a]
            E[s, a] = klambda * E[s, a]
        state= state_
        action = action_
      time = time + 1
    print("time:", time)
  for row in reversed(range(7)):
    for col in range(10):
      print(np.argmax(Q[pos_to_state([col, row]),:]), end=" ")
    print("")
  # print(Q)



if __name__ =='__main__':
  main()