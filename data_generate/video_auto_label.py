from multiprocessing import Process, Queue
import numpy as np
import cv2
from testing.mark_detector import MarkDetector
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def precess_video():

    cv2.namedWindow('face landmarks', cv2.WINDOW_NORMAL)
    vs = cv2.VideoCapture(VIDEO_PATH)
    time.sleep(2.0)

    mark_detector = MarkDetector(MODEL_FILE,CNN_INPUT_SIZE)
    if MODEL_FILE.split(".")[-1] == "pb":
        run_model = 0
    elif MODEL_FILE.split(".")[-1] == "hdf5" or MODEL_FILE.split(".")[-1] == "h5":
        run_model = 1
    else:
        print("input model format error !")
        return

    idx = 0
    while True:
        _, src = vs.read()
        if src is None:
            print("The video is precessed over !")
            break
        start = cv2.getTickCount()

        frame = src.copy()
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

                txt_filename = ("%06d" % idx) + ".pts"
                txt_file = os.path.join(OUTPUT_DIR, txt_filename)
                txt = open(txt_file, 'w')
                txt.write("version: 1\n")
                txt.write("n_points: %d\n" % marks.size)
                txt.write("{\n")
                for mark in marks:
                    txt.write("%.2f %.2f\n" % (mark[0], mark[1]))
                txt.write("}\n")
                txt.close()
                image_filename = ("%06d" % idx) + ".jpg"
                image_file = os.path.join(OUTPUT_DIR, image_filename)
                cv2.imwrite(image_file, src)

        fps_time = (cv2.getTickCount() - start)/cv2.getTickFrequency()
        cv2.putText(frame, '%.1ffps'%(1/fps_time), (frame.shape[1]-65,15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        # show the frame
        cv2.imshow("face landmarks", frame)

        key = cv2.waitKey(1)
        if key == ord("q") or key == 0xFF:
            break
        idx = idx + 1

    # do a bit of cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
   # MODEL_FILE = "../model/facial_landmark_MobileNet.h5"
   # MODEL_FILE = "../log/facial_landmark_MobileNet/smooth_L1-14-1.08050.hdf5"
    MODEL_FILE = "../model/facial_landmark_MobileNet.pb"
    VIDEO_PATH = "../data/IU.avi"
    OUTPUT_DIR = "../data/out"
    CNN_INPUT_SIZE = 64
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    precess_video()