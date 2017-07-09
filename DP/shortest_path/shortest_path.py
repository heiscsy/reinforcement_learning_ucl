from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

actions = [np.array([0,-1]),np.array([0,1]),
  np.array([-1,0]), np.array([1,0])]

def ValueIteration(value):
  value_new = np.zeros([4,4]);
  for row in range(4):
    for col in range(4):
      if row==0 and col ==0:
        value_new[row, col] = 0
      else:
        max_value = -1e8
        for act in actions:
          tar = np.array([row, col])+act
          if tar[0] in range(4) and \
            tar[1] in range(4) and \
            max_value<value[tar[0], tar[1]]-1:
            max_value=value[tar[0], tar[1]]-1
        value_new[row, col] = max_value
  return value_new

def main():
  value = np.zeros([4,4])
  iter = 0
  while True:
    print("Iter: ", iter)
    iter = iter+1
    value_new = ValueIteration(value)
    print(value_new)
    if np.sum(value - value_new)<1: 
      break
    value= value_new

if __name__=='__main__':
  main()