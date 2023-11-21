import cv2

from recognition_module import RecognitionModule

video = cv2.VideoCapture(0)
recognition_module = RecognitionModule(tolerance=0.575)

recognition_module.load_known_faces("known_faces")
recognition_module.load_known_barcodes("barcodes.csv")
recognition_module.detect_on_camera(video)
