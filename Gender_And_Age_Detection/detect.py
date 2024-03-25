import cv2
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def highlightFace(net, frame, conf_threshold=0.7):
    frame_copy = frame.copy()
    frame_height, frame_width = frame_copy.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return frame_copy, face_boxes

def processFace(frame, face_box, age_net, gender_net, MODEL_MEAN_VALUES, age_list, gender_list):
    padding = 20
    face = frame[max(0, face_box[1] - padding): min(face_box[3] + padding, frame.shape[0] - 1),
               max(0, face_box[0] - padding): min(face_box[2] + padding, frame.shape[1] - 1)]

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    return gender, age

def main(image_path):
    face_model_path = "opencv_face_detector.pbtxt"
    face_proto_path = "opencv_face_detector_uint8.pb"
    age_proto_path = "age_deploy.prototxt"
    age_model_path = "age_net.caffemodel"
    gender_proto_path = "gender_deploy.prototxt"
    gender_model_path = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']

    face_net = cv2.dnn.readNet(face_model_path, face_proto_path)
    age_net = cv2.dnn.readNet(age_model_path, age_proto_path)
    gender_net = cv2.dnn.readNet(gender_model_path, gender_proto_path)

    video = cv2.VideoCapture(image_path if image_path else 0)
    executor = ThreadPoolExecutor(max_workers=2)
    padding = 20
    while cv2.waitKey(1) < 0:
        has_frame, frame = video.read()
        if not has_frame:
            cv2.waitKey()
            break
        frame = cv2.flip(frame, 1)
        result_img, face_boxes = highlightFace(face_net, frame)
        if not face_boxes:
            print("No face detected")

        for face_box in face_boxes:
            gender, age = executor.submit(processFace, frame, face_box, age_net, gender_net, MODEL_MEAN_VALUES, age_list, gender_list).result()
            cv2.putText(result_img, f'{gender}, {age}', (face_box[0], face_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", result_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()
    main(args.image)
