from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from flask import Flask, render_template, Response

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
classifier = load_model("D:/FSDS/AI/Assignment/Assignment 4 - AI_Use_Case/cnn_model.h5")

class_labels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

cap = cv2.VideoCapture(0)

app = Flask(__name__)

def gen_frames():
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    
            if np.sum([roi_gray])!=0:
                roi=roi_gray.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)
                preds=classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position=(x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
        #resized_img = cv2.resize(frame, (1000, 700))  
            
        ret, buffer = cv2.imencode('.jpg', frame)
            
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)