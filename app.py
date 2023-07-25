from flask import Flask, render_template, request, Response,send_from_directory,session
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from Multipose_Image import runtensorflow
from Multipose_Video import runfunction_video
from Singlepose_Image import singlepose_image
from Singlepose_Video import singlepose_video

model_single = tf.saved_model.load('singlepose_thunder')
movenet_singlepose = model_single.signatures['serving_default'] 

model_multi = tf.saved_model.load('Multipose_lightning')
movenet_multipose = model_multi.signatures['serving_default'] 

app = Flask(__name__,template_folder='template_all')
app.config['IMAGE']='output_image'
app.secret_key = 'secret_for_route'
app.config['VIDEO']='output_video'
app.secret_key = 'secret_key_forwhat'
app.config['IMAGE_SINGLEPOSE']='output_image_singlepose'
app.config['VIDEO_SINGLEPOSE']='output_video_singlepose'

global img_text


#######################################################Multipose_webcam_Zone#############################################################3

# def runfunction(video_path):
# Optional if you are using a GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
def runfunction(movenet_multipose):
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

    threshold = .2

    cap = cv2.VideoCapture(0)
    w  = 640  # float `width`
    h = 640  # float `height`

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
        results = movenet_multipose(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

        for i in keypoints_with_scores:

            sum_confidence = sum(i[:10,2])/10
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
                    
                ret, buffer = cv2.imencode('.jpg',img_text)
                img_text = buffer.tobytes()

                
                    
            else:
                non_selected_keypoint = i


        


        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img_text + b'\r\n')
        
        # cv2.imshow('Movenet Multipose', img_text)
        
    cap.release()
    # img_text_resize = cv2.resize(img_text,(640,640))



@app.route("/loading_webcam", methods=['GET', 'POST'])
def loading():
    return render_template("newone_webcam.html")

@app.route('/livewebcam')
def video_webcam():
    return Response(runfunction(movenet_multipose),mimetype='multipart/x-mixed-replace; boundary=frame')

#######################################################singlepose_webcam_zone#############################################################

def runfunction_singlepose_webcam(movenet_singlepose):
    # # Download the model from TF Hub.
    # model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
    # model = tf.saved_model.load('singlepose_thunder')
    # movenet = model.signatures['serving_default']

    # Threshold for 
    threshold = .3

    # Loads video source (0 is for main webcam)
    
    cap = cv2.VideoCapture(0)

    # Checks errors while opening the Video Capture
    if not cap.isOpened():
        print('Error loading video')
        quit()


    success, img = cap.read()

    if not success:
        print('Error reding frame')
        quit()

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

        threshold_nose = keypoints[0][0][0][2]
        threshold_right_hand = keypoints[0][0][10][2]
        threshold_left_hand = keypoints[0][0][9][2]

        

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

                if threshold_nose < threshold or threshold_left_hand < threshold or threshold_right_hand < threshold:
                    
                    img_text = cv2.putText(img, 'No!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                
                else:
                
                    if nose > left_hand:
                        img_text = cv2.putText(img, 'Check!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (68,233,8), 1, cv2.LINE_AA)
                        # print('check!')
                    elif nose > right_hand:
                        img_text = cv2.putText(img, 'Check!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (68,233,8), 1, cv2.LINE_AA)
                        # print('check!')
                    else:
                        img_text = cv2.putText(img, 'No!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                        # print('No!')
        
        # # Shows image
        # cv2.imshow('Movenet', img_text)
        image_data = np.asarray(img_text)
        ret, buffer = cv2.imencode('.jpg',image_data)
        image_data = buffer.tobytes()


        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image_data + b'\r\n')
        
        # # Waits for the next frame, checks if q was pressed to quit
        # if cv2.waitKey(1) == ord("q"):
        #     break

        # Reads next frame
        success, img = cap.read()

    cap.release()

@app.route("/loading_webcam_singlepose", methods=['GET', 'POST'])
def loading_singlepose_webcam():
    return render_template("newone_singlepose_webcam.html")

@app.route('/livewebcam_singlepose')
def video_webcam_singlepose():
    return Response(runfunction_singlepose_webcam(movenet_singlepose),mimetype='multipart/x-mixed-replace; boundary=frame')

#######################################################firstpage_zone############################################################

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get('action1') == 'OpenWebcam':
            runfunction(movenet_multipose)
            return render_template("loading_webcam.html")
        elif request.form.get('action2') == 'Back':
            return render_template("index_webcam.html")
        elif request.form.get('action3') == 'Multipose':
            return render_template("Movenet_image_browser.html")
        elif request.form.get('action4') == 'Multipose':
            return render_template("Movenet_Video_browser.html")
        elif request.form.get('action5') == 'Multipose':
            return render_template("index_webcam.html")
        elif request.form.get('action6') == 'Singlepose':
            return render_template("singlepose_image.html")
        elif request.form.get('action7') == 'Singlepose':
            return render_template("singlepose_video.html")
        elif request.form.get('action8') == 'Singlepose':
            return render_template("index_singlepose_webcam.html")
        elif request.form.get('action9') == 'OpenWebcam':
            runfunction_singlepose_webcam(movenet_singlepose)
            return render_template("loading_singlepose_webcam.html")
        
        
    return render_template("homepage.html")


##############################IMAGEZONE############################################################
# @app.route('/', methods=['POST', 'GET'])
# def index():
#     return render_template("Movenet_image_browser.html")
    

@app.route('/loading_img', methods=['POST', 'GET'])
def loading_img():
    if 'get_image' not in request.files:
        error_message= 'Icutmyhairbecauseyoudontcaremyheart'
        return render_template('error.html', error_message=error_message)
    
    upload_folder="./input_image/"
    img = request.files['get_image']
    img_path = os.path.join(upload_folder, img.filename)
    img.save(img_path)

    session['get_image_from'] = img_path
    return render_template("loading_image.html")
   

@app.route('/classify_upload_img', methods=['POST', 'GET'])    
def runfunc_img():
    image_path = session.get('get_image_from',None)
    runtensorflow(image_path, movenet_multipose)
    result_image = 'outputing.jpg'
    return render_template("index_image.html",data=result_image)
    

@app.route('/image/<filename>')
def image(filename):
    return send_from_directory(app.config['IMAGE'], filename)

###############################################VIDEOZONE####################################################################

# @app.route('/', methods=['POST', 'GET'])
# def index():
#     return render_template("Movenet_Video_browser.html")

@app.route('/loading_video', methods=['POST', 'GET'])
def loading_video():
    if 'get_video' not in request.files:
        error_message= 'Icutmyhairbecauseyoudontcaremyheart'
        return render_template('error.html', error_message=error_message)
    
    upload_folder="./input_video/"
    vid = request.files['get_video']
    video_path = os.path.join(upload_folder, vid.filename)
    vid.save(video_path)

    session['get_video_from'] = video_path
    return render_template("loading_video.html")

@app.route('/classify_upload_video', methods=['POST', 'GET'])
def runfunc_video():
    
    video_path = session.get('get_video_from',None)
    runfunction_video(video_path, movenet_multipose)
    result_video = 'output.mp4'
    return render_template("index_video.html",data=result_video)
    

@app.route('/video/<filename>')
def video_video(filename):
    return send_from_directory(app.config['VIDEO'], filename)

##################################################################GOTOZONE#################################################

@app.route("/goto_home", methods=['GET', 'POST'])
def loading_home():
    return render_template("homepage.html")

@app.route("/goto_img", methods=['GET', 'POST'])
def goto_image():
    return render_template("Movenet_image_browser.html")

@app.route("/goto_video", methods=['GET', 'POST'])
def goto_video():
    return render_template("Movenet_Video_browser.html")

@app.route("/goto_webcam", methods=['GET', 'POST'])
def goto_webcam():
    return render_template("index_webcam.html")

@app.route("/goto_image_singlepose", methods=['GET', 'POST'])
def goto_singlepose_image():
    return render_template("singlepose_image.html")

@app.route("/goto_video_singlepose", methods=['GET', 'POST'])
def goto_video_singlepose():
    return render_template("singlepose_video.html")

@app.route("/goto_webcam_singlepose", methods=['GET', 'POST'])
def goto_webcam_singlepose():
    return render_template("index_singlepose_webcam.html")

###########################################singlepose_image_Zone#######################################################3
@app.route('/loading_img_single', methods=['POST', 'GET'])
def loading_img_single():
    if 'get_image' not in request.files:
        error_message= 'Icutmyhairbecauseyoudontcaremyheart'
        return render_template('error.html', error_message=error_message)
    
    upload_folder="./input_image_singlepose/"
    img = request.files['get_image']
    img_path = os.path.join(upload_folder, img.filename)
    img.save(img_path)

    session['get_image_from'] = img_path
    return render_template("loading_singlepose_image.html")
   

@app.route('/classify_upload_img_single', methods=['POST', 'GET'])    
def runfunc_img_single():
    image_path = session.get('get_image_from',None)
    singlepose_image(image_path, movenet_singlepose)
    result_image = 'outputing.jpg'
    return render_template("index_singlepose_image.html",data=result_image)
    

@app.route('/image_singlepose/<filename>')
def image_singlepose(filename):
    return send_from_directory(app.config['IMAGE_SINGLEPOSE'], filename)

##############################################singlepose_video_zone##################################################

@app.route('/loading_video_singlepose', methods=['POST', 'GET'])
def loading_video_singlepose():
    if 'get_video' not in request.files:
        error_message= 'Icutmyhairbecauseyoudontcaremyheart'
        return render_template('error.html', error_message=error_message)
    
    upload_folder="./input_video_singlepose/"
    vid = request.files['get_video']
    video_path = os.path.join(upload_folder, vid.filename)
    vid.save(video_path)

    session['get_video_from'] = video_path
    return render_template("loading_singlepose_video.html")

@app.route('/classify_upload_video_singlepose', methods=['POST', 'GET'])
def runfunc_video_singlepose():
    
    video_path = session.get('get_video_from',None)
    singlepose_video(video_path, movenet_singlepose)
    result_video = 'output.mp4'
    return render_template("index_singlepose_video.html",data=result_video)
    

@app.route('/video_singlepose/<filename>')
def video_video_singlepose(filename):
    return send_from_directory(app.config['VIDEO_SINGLEPOSE'], filename)


if __name__=="__main__":
    app.run(debug=True)
