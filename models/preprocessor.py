# Preprocess car racing frames to be grayscale and filter out the grass 
import gym
import numpy as np
import numpy.typing as npt
import cv2
from PIL import Image

"""
Process a given observation - expected to be (96, 96, 3)
and output a grayscale version (96, 96)

options:
crop the bottom cut off (84, 96, 3)

"""
def process_observation(observation: npt.ArrayLike, crop=False, randomized=False) -> npt.ArrayLike:
    # convert to grayscale
    output = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    if crop:
        output = output[0:84, :]

    if randomized:
        # TODO: detect road vs grass
        pass

    # threshold the image to turn all the grass into white
    # _, output = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY)
    # output = cv2.GaussianBlur(output, (5,5), 0)
    return output

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", continuous=False, new_step_api=True)
    state = env.reset()

    for _ in range(50):
        # no op
        state, *__ = env.step(0)

    cv2.imwrite("processed.png", process_observation(state))