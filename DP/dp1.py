from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

move = [np.array([1,0]), np.array([0,1]), np.array([-1,0]), np.array([0,-1])]
def legal_pose(pose):
    return pose[0]>=0 and pose[0]<4 and pose[1]>=0 and pose[1]<4

class Agent:
  def __init__(self):
    row = np.random.randint(low=0, high=4)
    col = np.random.randint(low=0, high=4)
    self.current = np.array([row, col])
  def move(self):
    move_idx = np.random.randint(low=0, high=4)
    move = [np.array([1,0]), np.array([0,1]), np.array([-1,0]), np.array([0,-1])]
    while not legal_pose(self.current + move[move_idx]):
      move_idx = np.random.randint(low=0, high=4)
    self.current = self.current+move[move_idx]


def policy_iteration(value, policy):
  value_new = np.zeros([4,4])
  for state in range(1, 15):
    row = state//4
    col = state % 4
    for m_idx in range(len(move)):
      m = move[m_idx]
      target_pose = np.array([row, col]) + m
      if legal_pose(target_pose):
        value_new[row, col] = policy[state-1, m_idx] * (value[target_pose[0], target_pose[1]]-1)+value_new[row, col]
      else:
        value_new[row, col] = policy[state-1, m_idx] * (value[row, col]-1)+value_new[row, col]
  return value_new

def update_policy(value, policy):
  policy_new = np.zeros([14,4])
  for state in range(1, 15):
    max_neightbour_value = -1000000
    max_neightbour_idx = []
    row = state//4
    col = state%4
    for m_idx in range(4):
      target_pose = np.array([row, col])+move[m_idx]
      target_value = value[target_pose[0], target_pose[1]] if\
        legal_pose(target_pose) else \
        value[row, col] 
      if target_value>max_neightbour_value:
        max_neightbour_value = target_value
        max_neightbour_idx = [m_idx]
      elif target_value==max_neightbour_value:
        max_neightbour_idx.append(m_idx)
    for m_idx in max_neightbour_idx:
      policy_new[state-1, m_idx] = 1/len(max_neightbour_idx)
  return policy_new

def visualize_value(value):
  print(value)

def visualize_policy(policy):
  print(policy)

def main():
  value = np.zeros([4,4])
  policy = np.ones([14,4]) * 0.25
  while True:
    idx=0
    while True:
      value_new = policy_iteration(value, policy)
      if np.sum((value-value_new)*(value-value_new))<0.1:
        break
      else:
        value = value_new
      print("step: %s"%(idx))
      visualize_value(value)
      idx = idx +1
    policy_new = update_policy(value, policy)
    if np.sum((policy-policy_new)*(policy-policy_new))<0.1:
     break
    else:
      policy=policy_new
    visualize_policy(policy)

if __name__ == '__main__':
  main()
