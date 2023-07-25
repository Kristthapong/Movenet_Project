# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

def runtensorflow(image_path):
    # # Download the model from TF Hub.
    model = tf.saved_model.load('singlepose_thunder')
    # model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
    movenet = model.signatures['serving_default']

    # # Threshold for 
    threshold = .3

    # Load the input image.
    # image_path = 'half.PNG'
    image = cv2.imread(image_path)



    y, x, _ = image.shape

    # # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
    tf_img = cv2.resize(image, (256,256))
    tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
    tf_img = np.asarray(tf_img)
    tf_img = np.expand_dims(tf_img,axis=0)

    # # Resize and pad the image to keep the aspect ratio and fit the expected size.
    images = tf.cast(tf_img, dtype=tf.int32)

    # # Run model inference.
    outputs = movenet(images)
    # # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']
    print(outputs)

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
            

    # # Shows image
    cv2.imwrite('./output/outputing.jpg', result)



# runtensorflow()
