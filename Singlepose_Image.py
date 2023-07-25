# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

def singlepose_image(img_path, movenet_singlepose):

    # # Download the model from TF Hub.
    # model = tf.saved_model.load('singlepose_thunder')
    # # model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
    # movenet = model.signatures['serving_default']

    # # Threshold for 
    threshold = .3


    # Load the input image.
    # image_path = 'half.PNG'
    image = cv2.imread(img_path)



    y, x, _ = image.shape

    # # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
    tf_img = cv2.resize(image, (256,256))
    tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
    tf_img = np.asarray(tf_img)
    tf_img = np.expand_dims(tf_img,axis=0)

    # # Resize and pad the image to keep the aspect ratio and fit the expected size.
    images = tf.cast(tf_img, dtype=tf.int32)

    # # Run model inference.
    outputs = movenet_singlepose(images)
    # # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']

    # print(outputs)

    # determine
    nose = keypoints[0][0][0][0]
    left_hand = keypoints[0][0][9][0]
    right_hand = keypoints[0][0][10][0]

    threshold_nose = keypoints[0][0][0][2]
    threshold_right_hand = keypoints[0][0][10][2]
    threshold_left_hand = keypoints[0][0][9][2]

    # # iterate through keypoints
    for k in keypoints[0,0,:,:]:
        # Converts to numpy array
        k = k.numpy()

    #     # Checks confidence for keypoint
        if k[2] > threshold:
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            yc = int(k[0] * y)
            xc = int(k[1] * x)

    #         # Draws a circle on the image for each keypoint
            result = cv2.circle(image, (xc, yc), 2, (255, 0, 0), 5)
            
            if nose > left_hand:
                img_text = cv2.putText(result, 'Check!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (68,233,8), 1, cv2.LINE_AA)
                # print('check!')
            elif nose > right_hand:
                img_text = cv2.putText(result, 'Check!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (68,233,8), 1, cv2.LINE_AA)
                # print('check!')
            elif threshold_nose < threshold:
                img_text = cv2.putText(result, 'No!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

            elif threshold_right_hand < threshold:
                img_text = cv2.putText(result, 'No!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

            elif threshold_left_hand < threshold:
                img_text = cv2.putText(result, 'No!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                # print('check!')
            else:
                img_text = cv2.putText(result, 'No!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                # print('No!')
        
            # img_text_resize = cv2.resize(img_text,(640,640))
    # # Shows image
    cv2.imwrite('./output_image_singlepose/outputing.jpg', img_text)



    # runtensorflow()
