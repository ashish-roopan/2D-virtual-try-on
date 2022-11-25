import os
import torch
import onnx
import onnxruntime as ort
import cv2
import numpy as np
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2


def preprocess_img(face_image, garment_image, transform):
    #hstack face and garment image
    face_image_bg = np.zeros_like(garment_image) 
    face_image_bg[0:face_image.shape[0], 0:face_image.shape[1]] = face_image
    full_image = np.hstack((face_image_bg, garment_image))

    # transform
    image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image = augmented['image']
    image = image.unsqueeze(0)
    return image, full_image

def merge_face_and_cloth(full_image, position):
    og_height = 1101
    og_width = 750
    position = position * np.array([og_width, og_height, og_width, og_height])
    x, y, w, h = position.astype(np.int)

    face_img = full_image[:h, :w]
    garment_img = full_image[:, og_width:]

    face_img_full = np.zeros_like(garment_img)

    face_img_full[y:y+h,x:x+w] = face_img

    face_alpha = face_img_full[:,:,3]
    face_fg = cv2.bitwise_and(face_img_full, face_img_full, mask=face_alpha)
    merged = cv2.add(garment_img,face_fg)
    return merged

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/model.onnx', help='model path')
    parser.add_argument('--data_dir', type=str, default='dataset/test_set/', help='data dir containing facehair and garment images')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    transform =  A.Compose([
                    A.Resize(256, 256),
                    A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
                    ToTensorV2()
                    ])

    #prepare model
    onnx_model = onnx.load(args.model_path)
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(args.model_path, providers=['CPUExecutionProvider'])

    dirs = os.listdir(args.data_dir)
    for data_dir in dirs:
        #prepare image
        face_img_path = os.path.join(args.data_dir, data_dir, 'facehair.png')
        garment_img_path = os.path.join(args.data_dir, data_dir, 'garment_top.png')
        face_img = cv2.imread(face_img_path, cv2.IMREAD_UNCHANGED)
        garment_img = cv2.imread(garment_img_path, cv2.IMREAD_UNCHANGED)

        img, full_image = preprocess_img(face_img, garment_img, transform)
    
        # predict
        with torch.no_grad():
            pred_position = ort_sess.run(None, {'input': img.cpu().numpy()})[0][0]
    
        print('pred_position', pred_position)
        #decode result
        pred_image = merge_face_and_cloth(full_image, pred_position)
        cv2.imshow('result', pred_image)
        cv2.waitKey(0)
    
    
    