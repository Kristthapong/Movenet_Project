# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

def singlepose_video(video_path, movenet_singlepose):
    # # Download the model from TF Hub.
    # model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
    # model = tf.saved_model.load('singlepose_thunder')
    # movenet = model.signatures['serving_default']

    # Threshold for 
    threshold = .3

    # Loads video source (0 is for main webcam)

    cap = cv2.VideoCapture(video_path)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    w  = cap.get(3)  # float `width`
    h = cap.get(4)  # float `height`

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output_video_singlepose/output.mp4',fourcc, 20.0, (int(w),int(h)))

    success, img = cap.read()

    y, x, _ = img.shape

    while success:
        # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
        tf_img = cv2.resize(img, (256,256))
        tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
        tf_img = np.asarray(tf_img)
        tf_img = np.expand_dims(tf_img,axis=0)

        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.cast(tf_img, dtype=tf.int32)

        # Run model inference.
        outputs = movenet_singlepose(image)
        # print(outputs)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints = outputs['output_0']

        # determine keypoint
        nose = keypoints[0][0][0][0]
        left_hand = keypoints[0][0][9][0]
        right_hand = keypoints[0][0][10][0]

        

        # iterate through keypoints
        for k in keypoints[0,0,:,:]:
            # Converts to numpy array
            k = k.numpy()

            # Checks confidence for keypoint
            if k[2] > threshold:
                # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
                yc = int(k[0] * y)
                xc = int(k[1] * x)

                # Draws a circle on the image for each keypoint
                img = cv2.circle(img, (xc, yc), 2, (0, 255, 0), 5)

                if nose > left_hand:
                    img_text = cv2.putText(img, 'Check!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (68,233,8), 1, cv2.LINE_AA)
                    # print('check!')
                elif nose > right_hand:
                    img_text = cv2.putText(img, 'Check!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (68,233,8), 1, cv2.LINE_AA)
                    # print('check!')
                else:
                    img_text = cv2.putText(img, 'No!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                    # print('No!')
        
        out.write(img_text)

        # Reads next frame
        success, img = cap.read()

    cap.release()