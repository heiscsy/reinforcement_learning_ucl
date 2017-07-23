from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from pynput.keyboard import Key, Listener

import gym
import numpy as np

kEpisode = 9000
kEpsilon = 0.5
kAlpha = 0.1
kGammar = 1
action_list = [[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]

example_num = 10

action = 1

def on_press(key):
    global action
    if key==Key.left:
        action = 0
    elif key==Key.right:
        action = 2

def on_release(key):
    global action
    action = 1
    if key == Key.esc:
        # Stop listener
        return False

def main():
  global action
  sars = []
  # w = (np.random.random([5])-0.5)*20
  env = gym.make('MountainCar-v0')
  with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    eps = 0
    while eps<example_num:
      space = env.reset()
      count = 0
      while True:
        env.render()
        space_, reward, done, _ =env.step(action)
        sars.append([space, action, reward, space_])
        count = count + 1
        if done:
          print(reward)
          print(count)
          break
      if count<200:
        eps = eps + 1
    print(sars)
    listener.join()




if __name__=='__main__':
  main()