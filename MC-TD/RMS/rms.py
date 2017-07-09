from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

kGamma = 1

kAlphaTD = 0.1
kAlphaMC = 0.01

def init():
  state = 2
  return state

def play(state):
  terminate = False
  reward = 0
  action=np.random.randint(low=0, high=2)
  action=-1 if action ==0 else 1
  new_state = state+action
  if new_state < 0:
    terminate = True
    reward = 0
  elif new_state > 4:
    terminate = True
    reward = 1
  else:
    terminate = False
    reward = 0
  return terminate, reward, new_state

def main():
  value_td = np.ones(5) * 0.5
  value_mc = np.ones(5) * 0.5
  state_count = np.zeros(5)
  for _ in range(100):
    terminate = False
    state_trail = []
    state = init()
    state_trail.append(state)
    while not terminate:
      terminate, reward, state_new = play(state)
      if not terminate:
        value_td[state] = value_td[state] + kAlphaTD*(reward + value_td[state_new] 
          * kGamma - value_td[state])
        state_trail.append(state_new)
        state = state_new
      else:
        value_td[state] = value_td[state] + kAlphaTD*(reward  - value_td[state])
    for s in reversed(state_trail):
      value_mc[s] = value_mc[s] + kAlphaMC*(reward - value_mc[s])
  print('TD:', value_td)
  print("-------------")
  print('MC:', value_mc)

if __name__ == '__main__':
  main()