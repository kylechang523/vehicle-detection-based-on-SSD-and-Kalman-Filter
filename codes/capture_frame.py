#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:07:09 2016

@author: kyleguan
A simple script to capture frames from a video clip

"""

import cv2
import os

video_name='part2.mp4'
start_time=0
end_time=49000
step=40

dir_name ='part2/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

vidcap=cv2.VideoCapture(video_name)
for i, time in enumerate(range(start_time, end_time, step)):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time)
    success, image=vidcap.read()
    if success:
       # Need to create the directory ( 'highway') first 
       file_name='part2/frame{:03d}.jpg'.format(i+1)
       cv2.imwrite(file_name,image)
   