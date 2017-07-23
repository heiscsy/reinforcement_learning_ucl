from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import keyboard
import time

if __name__=="__main__":
  # action = 1
  # pygame.init()
  # pygame.display.set_mode()

  while True:
    # keys=pygame.key.get_pressed()
    # if keys[pygame.K_LEFT]:
    #   action = 0
    # elif keys[pygame.K_RIGHT]:
    #   action = 2
    # # else:
    #   # action = 1
    # print(action)
    # time.sleep(0.1)
    # events = pygame.event.get()
    # for event in events:
    #     print(event)
    #     if event.type == pygame.KEYDOWN:
    #         print(event.key)
    #         if event.key == pygame.K_LEFT:
    #             action = 0
    #         if event.key == pygame.K_RIGHT:
    #             action = 2
    # # print(action)
    if keyboard.is_pressed('a'): #if key 'a' is pressed 
      print('You Pressed A Key!')