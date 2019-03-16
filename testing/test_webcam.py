from multiprocessing import Process, Queue
import numpy as np
import cv2
from testing.mark_detector import MarkDetector
import time
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


def webcam_main():
    print("Camera sensor warming up...")
    cv2.namedWindow('face landmarks', cv2.WINDOW_NORMAL)
    vs = cv2.VideoCapture(VIDEO_PATH)
    time.sleep(2.0)

    mark_detector = MarkDetector(current_model, CNN_INPUT_SIZE)
    if current_model.split(".")[-1] == "pb":
        run_model = 0
    elif current_model.split(".")[-1] == "hdf5" or current_model.split(".")[-1] == "h5":
        run_model = 1
    else:
        print("input model format error !")
        return

    # loop over the frames from the video stream
    while True:
        _, frame = vs.read()
        if frame is None:
            break
        start = cv2.getTickCount()

       # frame = cv2.resize(frame, (750,750))
       # frame = cv2.flip(frame, 1)
        faceboxes = mark_detector.extract_cnn_facebox(frame)

        if faceboxes is not None:
            for facebox in faceboxes:
                # Detect landmarks from image of 64X64 with grayscale.
                face_img = frame[facebox[1]: facebox[3], facebox[0]: facebox[2]]
                cv2.rectangle(frame, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0, 255, 0), 2)
                face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_img0 = face_img.reshape(1, CNN_INPUT_SIZE, CNN_INPUT_SIZE, 1)

                if run_model == 1:
                    marks = mark_detector.detect_marks_keras(face_img0)
                else:
                    marks = mark_detector.detect_marks_tensor(face_img0, 'input_2:0', 'output/BiasAdd:0')
                # marks *= 255
                marks *= facebox[2] - facebox[0]
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]
                # Draw Predicted Landmarks
                mark_detector.draw_marks(frame, marks, color=(255, 255, 255), thick=2)

        fps_time = (cv2.getTickCount() - start)/cv2.getTickFrequency()
        cv2.putText(frame, '%.1ffps'%(1/fps_time), (frame.shape[1]-65,15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        # show the frame
        cv2.imshow("face landmarks", frame)
        # writer.write(frame)
        key = cv2.waitKey(1)

        # if the `q` key was pressed, break from the loop
        if key == ord("q") or key == 0xFF:
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
#    current_model = "../model/facial_landmark_cnn.h5"
    current_model = "../model/facial_landmark_SqueezeNet.pb"
    VIDEO_PATH = "../data/vid.avi"
#    VIDEO_PATH = 0
    CNN_INPUT_SIZE = 64
    webcam_main()