from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def face_emo():
    face_classifier = cv2.CascadeClassifier(r'D:\Research\DL Depression\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
    classifier = load_model(r"D:\Research\DL Depression\Emotion_Detection_CNN-main\model.h5")

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    labels = {}

    for i in emotion_labels:
        labels[i]=0

    # Specify the path to your video file
    video_path = r"D:\Research\DL Depression\EmoRec\backend\video.webm"

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # Break the loop if there are no more frames
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                labels[label]+=1
        #         label_position = (x, y)
        #         cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #     else:
        #         cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.imshow('Emotion Detector', frame)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    

    cap.release()

    cv2.destroyAllWindows()

    return (max(labels, key = labels.get)), labels

# face_emo()