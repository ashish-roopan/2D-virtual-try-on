import json
import cv2
import numpy as np
import os


dirs = os.listdir('./dataset')

#check positions.json
for dir in dirs:
    positions_path = './dataset/' + dir + '/positions.json'
    positions = json.load(open(positions_path))
    x = positions['x']
    y = positions['y']
    w = positions['w']
    h = positions['h']

    positions = [x,y,w,h]
    print(positions)
    if x < 0 or y < 0 or w < 0 or h < 0:
        print('error1')
        break

    if x + w > 750 or y + h > 1101:
        print('error2')
        break

    if w > 750 or h > 1101:
        print('error3')
        break

    if len(positions) != 4:
        print('error4')
        break

#check facehair.png
for dir in dirs:
    facehair_path = './dataset/' + dir + '/facehair.png'
    facehair_img = cv2.imread(facehair_path,cv2.IMREAD_UNCHANGED)
    if facehair_img is None:
        print('error5')
        break
    
    if facehair_img.shape[0] > 1101 or facehair_img.shape[1] > 750:
        print('error6')
        break
    
    if len(facehair_img.shape) != 3:
        print('error7')
        break

    height, width, channels = facehair_img.shape
    if channels != 4:
        print('error8')
        break

    if height == 0 or width == 0:
        print('error9')
        break

#check garment_top.png
for dir in dirs:
    garment_path = './dataset/' + dir + '/garment_top.png'
    garment_img = cv2.imread(garment_path,cv2.IMREAD_UNCHANGED)
    if garment_img is None:
        print('error10')
        break
    
    if garment_img.shape[0] > 1101 or garment_img.shape[1] > 750:
        print('error11')
        break
    
    if len(garment_img.shape) != 3:
        print('error12')
        break

    height, width, channels = garment_img.shape
    if channels != 4:
        print('error13')
        break

    if height == 0 or width == 0:
        print('error14')
        break
