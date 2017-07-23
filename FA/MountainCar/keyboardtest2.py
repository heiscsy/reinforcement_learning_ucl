from __future__ import print_function
from pynput.keyboard import Key, Listener

import time
state = 1

def on_press(key):
    global state
    if key==Key.left:
        state = 0
    elif key==Key.right:
        state = 2

def on_release(key):
    global state
    state = 1
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    while True:
        time.sleep(0.1)
        print(state)
    listener.join()