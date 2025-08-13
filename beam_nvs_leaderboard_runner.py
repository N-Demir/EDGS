# You shouldn't need to edit this file, but feel free to take a look at how things are called and run remotely
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import modal
from beam_nvs_leaderboard_image import image

from beam import function

@function(image=image, gpu="RTX4090")
def hello_world():
    print("Hello, world!")

hello_world()