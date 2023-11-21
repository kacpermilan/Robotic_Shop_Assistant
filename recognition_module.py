import torch
import face_recognition
import os
import pickle
import cv2
import csv
import numpy as np
from pyzbar.pyzbar import decode

del torch


# Notatka do mnie z przyszłości.
# Apparently CUDA nie jest wykrywana jak instalujesz dlib-a przez pip-a, więc musisz to zbudować customowo:
# https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781
class RecognitionModule:
    def __init__(self, model="cnn", tolerance=0.5,
                 frame_thickness=2, font_thickness=2):
        self.model = model
        self.tolerance = tolerance
        self.frame_thickness = frame_thickness
        self.font_thickness = font_thickness
        self.known_faces_encodings = []
        self.known_names = []
        self.known_barcodes = {}

    def load_known_faces(self, known_faces_dir, force_rebuild: bool = False):
        """
        Load known faces encodings from a faces directory's cache file, or create a new cache based on its contents.
        """
        print("Loading known faces...")
        cache_file = os.path.join(known_faces_dir, 'face_encodings_cache.pkl')

        if os.path.exists(cache_file) and not force_rebuild:
            print("Loading from cache...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.known_faces_encodings = cache_data['encodings']
                self.known_names = cache_data['names']
                return

        print("Building the cache...")
        self.create_face_encodings(known_faces_dir, cache_file)

    def load_known_barcodes(self, csv_file_path):
        """
        Load known barcodes from a CSV file into a dictionary.

        The CSV file should have two columns: barcode data and the associated name or information.
        """
        try:
            with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row:
                        barcode_data, associated_info = row
                        self.known_barcodes[barcode_data] = associated_info
        except FileNotFoundError:
            print(f"File not found: {csv_file_path}")
        except Exception as e:
            print(f"Error loading known barcodes: {e}")

    def create_face_encodings(self, known_faces_dir, cache_file: str):
        """
        Build a new face encodings, based on the faces found in the given directory, and store them in the cache file.
        """
        for name in os.listdir(known_faces_dir):
            directory_path = os.path.join(known_faces_dir, name)
            if not os.path.isdir(directory_path):
                continue

            for filename in os.listdir(directory_path):
                image = face_recognition.load_image_file(os.path.join(directory_path, filename))
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    encoding = encodings[0]
                    self.known_faces_encodings.append(encoding)
                    self.known_names.append(name)
                else:
                    print(f"No faces found in the image: {filename}")

        with open(cache_file, 'wb') as f:
            pickle.dump({'encodings': self.known_faces_encodings, 'names': self.known_names}, f)

    def detect_on_camera(self, video: cv2.VideoCapture):
        """
        Try to detect the known faces (and all the rest) using the given VideoCapture instance.
        """
        print("Detecting on camera...")
        while True:
            ret, image = video.read()
            locations = face_recognition.face_locations(image, model=self.model)
            encodings = face_recognition.face_encodings(image, locations)

            self.__recognize_faces(image, locations, encodings)
            self.__detect_barcodes(image)

            cv2.imshow("Camera Video", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def test_on_unknown_faces(self, test_dir: str):
        """
        Test the currently stored known faces encodings with the images in a given test directory.
        """
        print("Evaluating against unknown faces...")
        for filename in os.listdir(test_dir):
            print(filename)
            image = face_recognition.load_image_file(os.path.join(test_dir, filename))
            locations = face_recognition.face_locations(image, model=self.model)
            encodings = face_recognition.face_encodings(image, locations)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            self.__recognize_faces(image, locations, encodings)

            cv2.imshow(filename, image)
            cv2.waitKey(0)
            cv2.destroyWindow(filename)

    def test_on_barcode_images(self, test_dir: str):
        """
        Test the barcode detection against the images in a given test directory.
        """
        print("Evaluating against barcode images...")
        for filename in os.listdir(test_dir):
            print(filename)
            image = cv2.imread(os.path.join(test_dir, filename))

            self.__detect_barcodes(image)

            cv2.imshow(filename, image)
            cv2.waitKey(0)
            cv2.destroyWindow(filename)

    def __recognize_faces(self, image, locations, encodings):
        """
        Compare the encodings of the faces found on the image with the known faces encodings and mark all faces.
        """
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(self.known_faces_encodings, face_encoding, self.tolerance)
            if True in results:
                label = self.known_names[results.index(True)]
                color = self.__name_to_color(label)
                print(f"Match found: {label}")
            else:
                label = "Generic Human"
                color = [0, 255, 0]

            darker_color = self.__darken_color(color)
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            cv2.rectangle(image, top_left, bottom_right, color, self.frame_thickness)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, darker_color, cv2.FILLED)
            cv2.putText(image, label, (face_location[3] + 10, face_location[2] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), self.font_thickness)

    def __detect_barcodes(self, image):
        color = [255, 0, 0]
        darker_color = self.__darken_color(color)
        results = decode(image)
        for decoded_barcode in results:
            if decoded_barcode.type == 'QRCODE':
                continue

            decoded_data = decoded_barcode.data.decode()
            label = self.known_barcodes.get(decoded_data, decoded_data)

            top_left = (decoded_barcode.rect.left, decoded_barcode.rect.top)
            bottom_right = (decoded_barcode.rect.left + decoded_barcode.rect.width,
                            decoded_barcode.rect.top + decoded_barcode.rect.height)
            cv2.rectangle(image, top_left, bottom_right, color, self.frame_thickness)

            cv2.polylines(image, [np.array(decoded_barcode.polygon)], True, (0, 255, 0), 2)

            top_left = (decoded_barcode.rect.left + decoded_barcode.rect.width,
                        decoded_barcode.rect.top)
            bottom_right = (decoded_barcode.rect.left,
                            decoded_barcode.rect.top - 22)
            cv2.rectangle(image, top_left, bottom_right, darker_color, cv2.FILLED)
            cv2.putText(image, label,(decoded_barcode.rect.left + 10, decoded_barcode.rect.top - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), self.font_thickness)

    @staticmethod
    def __name_to_color(string: str) -> list[int]:
        """
        Take 3 first letters, tolower()
        lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
        :param string: String, which first 3 letters will be used for picking colour
        :return: RGB array of a colour
        """

        return [(ord(c.lower()) - 97) * 8 for c in string[:3]]

    @staticmethod
    def __darken_color(old_color: list[int], factor: float = 0.5) -> tuple[int]:
        """
        Darken a given color by a factor.
        Factor should be between 0 and 1, where 0 is black and 1 is the original color.
        """
        return tuple(max(0, int(c * factor)) for c in old_color)
