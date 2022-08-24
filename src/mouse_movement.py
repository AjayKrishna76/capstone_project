# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 00:26:13 2022

@author: ajayk
"""
#!pip install pyautogui
import pyautogui
import numpy as np
import time

print(pyautogui.size())

while True:
    w = np.random.randint(0,1919)
    h = np.random.randint(0,1079)
    pyautogui.moveTo(w,h, duration = 1)
    time.sleep(10)
