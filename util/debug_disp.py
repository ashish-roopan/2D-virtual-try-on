import cv2
import numpy as np
import torch



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

def debug_disp(model, dataloader, device):
    full_image, image, gt_position = next(iter(dataloader))

    image = image.to(device)
    gt_position = gt_position.to(device)
    model.to(device)

    model.eval()
    with torch.no_grad():
        pred_position = model(image)

    full_image = full_image[0].cpu().numpy()
    gt_position = gt_position[0].cpu().numpy()
    pred_position = pred_position[0].cpu().numpy()

    gt_image = merge_face_and_cloth(full_image, gt_position)
    pred_image = merge_face_and_cloth(full_image, pred_position)
    
    image = np.hstack((gt_image, pred_image))

    return image



mean = np.array([0.406, 0.456, 0.485])
sd = np.array([0.225, 0.224, 0.229])