import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np

from flask import Flask, render_template, request, Response,send_from_directory,session



EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


model = tf.saved_model.load('Multipose_lightning')
movenet = model.signatures['serving_default']


# frame = cv2.imread('test4.PNG')
# frame = cv2.resize(frame,(640,640))
# y, x, _ = frame.shape


threshold = .1

cap = cv2.VideoCapture(0)
w  = 640  # float `width`
h = 640 # float `height`

while cap.isOpened():

    success, frame = cap.read()
    
    if frame is None:
        break
    
    frame = cv2.resize(frame,(640,640))
    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
    input_img = tf.cast(img, dtype=tf.int32)

    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

    for i in keypoints_with_scores:

        sum_confidence = sum(i[:11,2])/11
        print(sum_confidence)

        if sum_confidence > threshold:
            selected_keypoint = i
            y, x, c = frame.shape
            shaped = np.squeeze(np.multiply(selected_keypoint, [y,x,1]))

            nose = selected_keypoint[0]
            right_hand_joint = selected_keypoint[10]
            left_hand_joint = selected_keypoint[9]

            nx, ny, np_confi = nose
            rx, ry, rp_confi = right_hand_joint
            lx, ly, lp_confi = left_hand_joint
        

            # print(right_hand_joint, left_hand_joint)
            # print(rx, ry)
            # print(lx, ly)

            for edge, color in EDGES.items():
                p1, p2 = edge
                y1, x1, c1 = shaped[p1]
                y2, x2, c2 = shaped[p2]
            
                if (c1 > threshold) & (c2 > threshold):      
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

            for kp in shaped:
                ky, kx, kp_conf = kp
                

                if kp_conf > threshold:
                    cv2.circle(frame, (int(kx), int(ky)), 5, (0,255,0), -1)

        
                

            if nx > rx:
                img_text = cv2.putText(frame, 'Check!', (int(ry*h), int(rx*w)), cv2.FONT_HERSHEY_SIMPLEX, 1, (68,233,8), 1, cv2.LINE_AA)
                
                # print(img_text)


            elif nx > lx:
                img_text = cv2.putText(frame, 'Check!', (int(ly*h), int(lx*w)), cv2.FONT_HERSHEY_SIMPLEX, 1, (68,233,8), 1, cv2.LINE_AA)
                
                # print(img_text)
                
                    
            else:
                img_text = cv2.putText(frame, 'No!', (int(ny*h), int(nx*w)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                
    
            
                
        else:
            non_selected_keypoint = i

   
            
    
    
cap.release()
# img_text_resize = cv2.resize(img_text,(640,640))









