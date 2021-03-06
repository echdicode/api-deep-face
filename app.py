import Flask, jsonify, request, render_template
import numpy as np
from keras.models import load_model as ld
from keras.preprocessing import image as imm
import cv2
import mtcnn
detector = mtcnn.MTCNN()
emotion_model = ld('model/model.h5')
class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
app = Flask(__name__)
app.config['SECRET_KEY'] = 'apikeyplayemo0'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
@app.route("/", methods=['POST', 'GET'])
def index():
    if (request.method == 'POST'):
        try:
            image_obj = request.files.get(key='image')
            img = image_obj.read()
            image = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_UNCHANGED)
            face = detect_face_mtcnn(image)
            if face is True:
                result = emotion_prediction(emotion_model, 'temp/capture.jpg')
                return jsonify({'result': result})

            else:

                return jsonify({'result': 'No Face or More than 1(One) face in Frame'})

        except Exception as e:

            return jsonify({"Error": str(e)})
    else:
        return "thanh cong"
def detect_face_mtcnn(img):
    try:
        detected_faces = detector.detect_faces(img)
        if len(detected_faces) == 1:
            face = detected_faces[0]
            x, y, width, height = face['box']
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            detected_face_crop = img[y:y + height, x:x + width]
            cv2.imwrite('temp/capture.jpg', detected_face_crop)
            return True

        else:

            return False

    except Exception as e:

        return str(e)
def emotion_prediction(emotion_model, image_path):
    img = imm.load_img(image_path, color_mode="grayscale", target_size=(48, 48))
    img_array = imm.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255
    custom = emotion_model.predict(img_array)
    prediction = class_labels[custom[0].argmax()]
    return prediction
if __name__ == '__main__':
    app.run(debug=True)



