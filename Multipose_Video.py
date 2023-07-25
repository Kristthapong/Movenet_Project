import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np

def runfunction_video(video_path, movenet_multipose):
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

    threshold = 0.2


    # model = tf.saved_model.load('Multipose_lightning')
    # movenet = model.signatures['serving_default']



    cap = cv2.VideoCapture(video_path)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    w  = 640 # float `width`
    h = 640  # float `height`

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output_video/output.mp4',fourcc, 20.0, (int(w),int(h)))

    while cap.isOpened():
        success, frame = cap.read()


        if frame is None:
            break

        frame = cv2.resize(frame,(640,640))

        # Resize image
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 224,224)
        input_img = tf.cast(img, dtype=tf.int32)
        
        # Detection section
        results = movenet_multipose(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        # print(keypoints_with_scores)
        for i in keypoints_with_scores:

            sum_confidence = sum(i[:,2])/17
            # print(sum_confidence)

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

                # out.write(cap)
        # Render keypoints 
        
        
        out.write(img_text)
        # cv2.imshow('Movenet Multipose', img_text)
        
    cap.release()
    cv2.destroyAllWindows()

