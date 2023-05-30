from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

import cv2
import dlib
import numpy as np

def img_to_landmark(img_path, detector, processor):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,1)
    
    top_left = rects[0].tl_corner()
    top_right = rects[0].br_corner()
    
    # print('rects:', rects)
    # print('top_left:' ,top_left)
    # print('top_right:', top_right)
    # print()
    
    x0 = top_left.x
    y0 = top_left.y
    w = rects[0].width()
    h = rects[0].height()
    print(x0)
    print(y0)
    print(w)
    print(h)
    bbox = [x0*1.0, y0*1.0, w, h]
    features = processor.inference(image, [bbox])
    landmarks = np.array(features['landmarks'][0])
    headpose = np.array(features['headpose'][0])

    return rects, landmarks.shape, headpose


detector = dlib.get_frontal_face_detector()
dataset = 'wflw'
processor = SPIGAFramework(ModelConfig(dataset))
print(img_to_landmark('/home2/dataset/MELD/train/dia0/utt0/000003.jpg', detector, processor))    