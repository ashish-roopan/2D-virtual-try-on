import os
import json
import cv2 
import numpy as np


def merge_alpha(garment_top_img,facehair_img,x,y,w,h):

    facehair_full = np.zeros_like(garment_top_img)
    facehair_full[y:y+h,x:x+w] = facehair_img

    facehair_alpha = facehair_full[:,:,3]

    facehair_fg = cv2.bitwise_and(facehair_full,facehair_full,mask=facehair_alpha)
    merged = cv2.add(garment_top_img,facehair_fg)
    return merged


def generate_merged_image(input_path):
    facehair_path = input_path + '/facehair.png'
    garment_top_path = input_path + '/garment_top.png'
    positions_path = input_path + '/positions.json'
    facehair_img = cv2.imread(facehair_path,cv2.IMREAD_UNCHANGED)
    garment_top_img = cv2.imread(garment_top_path,cv2.IMREAD_UNCHANGED)
    positions = json.load(open(positions_path))

    # print('facehair_img.shape',facehair_img.shape)
    x = positions['x'] + 1
    y = positions['y'] + 1
    w = positions['w']
    h = positions['h']

    merged = merge_alpha(garment_top_img,facehair_img,x,y,w,h)
    output_img = garment_top_img.copy()
    output_img[y:y+h,x:x+w] = facehair_img
    return output_img, facehair_img, garment_top_img, merged
    
dirs = os.listdir('./dataset')

for dir in dirs:
    input_path = './dataset/' + dir
    output_img, facehair_img, garment_top_img, merged_img = generate_merged_image(input_path)
    # cv2.imshow('merged', output_img)
    # cv2.imshow('facehair', facehair_img)
    # cv2.imshow('garment_top', garment_top_img)
    # cv2.imshow('merged2', merged_img)

    
    h, w, c = output_img.shape
    if h != 1101 or w != 750:
        print('input_path',input_path, facehair_img.shape, garment_top_img.shape, output_img.shape)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.destroyAllWindows()


