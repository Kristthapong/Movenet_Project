import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np

def runfunction_video(video_path, movenet_multipose):
    # Optional if you are using a GPU
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

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




    # model = tf.saved_model.load('Multipose_lightning')
    # movenet = model.signatures['serving_default']

    def draw_keypoints(frame, keypoints, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
        
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

    def draw_connections(frame, keypoints, edges, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
        
        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            
            if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

    # Function to loop through each person detected and render
    def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
        for person in keypoints_with_scores:
            draw_connections(frame, person, edges, confidence_threshold)
            draw_keypoints(frame, person, confidence_threshold)

    cap = cv2.VideoCapture(video_path)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    w  = cap.get(3)  # float `width`
    h = cap.get(4)  # float `height`

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output_video/output.mp4',fourcc, 20.0, (int(w),int(h)))

    while cap.isOpened():
        success, frame = cap.read()


        if frame is None:
            break

        # Resize image
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
        input_img = tf.cast(img, dtype=tf.int32)
        
        # Detection section
        results = movenet_multipose(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        # print(keypoints_with_scores)
        
        # Render keypoints 
        loop_through_people(frame, keypoints_with_scores, EDGES, 0.3)
        
        out.write(frame)

        # cv2.imshow('Movenet Multipose', frame)
        
    cap.release()
    cv2.destroyAllWindows()

