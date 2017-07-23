from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

class GridWorld:
  start = [0, 0]
  goal = [0, 11]
  action = [[-1,0], [0,1], [1,0], [0,-1]]

  def __init__(self):
    self.init()

  def init(self):
    self.current = self.start
    return self.current

  def confine_range(self, new_pos):
    if new_pos[0] < 0:
      new_pos[0] = 0
    if new_pos[0] > 3:
      new_pos[0] = 3
    if new_pos[1] < 0:
      new_pos[1] = 0
    if new_pos[1] > 11:
      new_pos[1] = 11
    return new_pos

  def drop(self, pos):
    drop = False
    if pos[0]==0 and pos[1] in range(1, 11):
      pos = [0, 0]
      drop = True
    return drop, pos

  def play(self, action):
    terminate = False
    new_pos = np.array(self.current) + np.array(self.action[action])
    self.current = [new_pos[0], new_pos[1]]
    self.current = self.confine_range(self.current)
    drop, self.current = self.drop(self.current)
    reward = -1
    if drop:
      reward = -100
      termintate = True
    if self.goal[0] == new_pos[0] and \
        self.goal[1]==new_pos[1]:
      terminate = True
    return self.current, reward, terminate
