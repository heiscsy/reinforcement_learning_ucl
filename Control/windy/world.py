from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

class GridWorld:
  wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
  start = [0, 3]
  goal = [7, 3]
  action = [[-1,0], [0,1], [1,0], [0,-1]]

  def __init__(self):
    self.init()

  def init(self):
    self.current = self.start
    return self.current

  def play(self, action):
    terminate = False
    new_pos = np.array(self.current) + np.array(self.action[action])
    if new_pos[0] < 0:
      new_pos[0] = 0
    if new_pos[0] > 9:
      new_pos[0] = 9
    new_pos = new_pos + np.array([0, self.wind[new_pos[0]]])
    if new_pos[1] < 0:
      new_pos[1] = 0
    if new_pos[1] > 6:
      new_pos[1] = 6
    self.current = [new_pos[0], new_pos[1]]
    if self.goal[0] == new_pos[0] and \
        self.goal[1]==new_pos[1]:
      reward=0
      terminate = True
    else:
      reward=-1
    return self.current, reward, terminate
