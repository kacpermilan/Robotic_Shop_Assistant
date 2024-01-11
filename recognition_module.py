import torch
import face_recognition
import os
import pickle
import cv2
import numpy as np
from pyzbar.pyzbar import decode
del torch


# Notatka do mnie z przyszłości.
# Apparently CUDA nie jest wykrywana jak instalujesz dlib-a przez pip-a, więc musisz to zbudować customowo:
# https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781
class RecognitionModule:
    """
        The `RecognitionModule` class provides functionality for face recognition and barcode detection.

        Args:
            model (str): The model to use for face recognition. Default is "cnn".
            tolerance (float): The tolerance value for face recognition. Default is 0.5.
            frame_thickness (int): The thickness of the frame when marking faces and barcodes. Default is 2.
            font_thickness (int): The thickness of the font when labeling faces and barcodes. Default is 2.

        Attributes:
            model (str): The model used for face recognition.
            tolerance (float): The tolerance value for face recognition.
            frame_thickness (int): The thickness of the frame when marking faces and barcodes.
            font_thickness (int): The thickness of the font when labeling faces and barcodes.
            known_faces_encodings (list): List of known faces encodings.
            known_names (list): List of known face names.

        Methods:
            load_known_faces(known_faces_dir, force_rebuild=False):
                Load known faces encodings from a faces directory's cache file,
                or create a new cache based on its contents.
            load_known_barcodes(csv_file_path):
                Load known barcodes from a CSV file into a dictionary.
            create_face_encodings(known_faces_dir, cache_file):
                Build a new face encodings, based on the faces found in the given directory,
                and store them in the cache file.
            detect_on_camera(video): Try to detect the known faces (and all the rest) using the
                                     given VideoCapture instance.
            test_on_unknown_faces(test_dir):
                Test the currently stored known faces encodings with the images in a given test directory.
            test_on_barcode_images(test_dir): Test the barcode detection against the images in a given test directory.
    """
    def __init__(self, model="cnn", tolerance=0.5,
                 frame_thickness=2, font_thickness=2, font_size=0.6):
        self.model = model
        self.tolerance = tolerance
        self.frame_thickness = frame_thickness
        self.font_thickness = font_thickness
        self.font_size = font_size

        self.product_data_source = None

        self.known_faces_encodings = []
        self.known_names = []

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

    def set_product_data_source(self, database_context):
        """
        Set the product data source, usually database module.
        """
        self.product_data_source = database_context

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

    def detect_faces(self, image):
        """
        Compare the encodings of the faces found on the image with the known faces encodings and save their positions.
        """
        locations = face_recognition.face_locations(image, model=self.model)
        encodings = face_recognition.face_encodings(image, locations)
        detected_faces = []

        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(self.known_faces_encodings, face_encoding, self.tolerance)
            if True in results:
                label = self.known_names[results.index(True)]
                print(f"Match found: {label}")
            else:
                label = "Customer"

            detected_faces.append((face_location, label))

        return detected_faces

    def detect_products(self, image):
        """
        Compare the barcodes found on the image with the database and save their positions.
        """
        results = decode(image)
        recognized_products = []

        for decoded_barcode in results:
            if decoded_barcode.type == 'QRCODE':
                continue

            decoded_data = decoded_barcode.data.decode()
            product_id = None

            if self.product_data_source is not None:
                product_id = self.product_data_source.barcodes.get(decoded_data)

            recognized_products.append((decoded_barcode, product_id))

        return recognized_products

    def display_detected_objects(self, image, detected_faces=None, detected_barcodes=None):
        if detected_barcodes is None:
            detected_barcodes = []

        if detected_faces is None:
            detected_faces = []

        for face, label in detected_faces:
            self.__mark_face(image, face, label)

        for barcode, product_id in detected_barcodes:
            if product_id is not None:
                product = self.product_data_source.products.get(product_id)
                label = f"{product['name']}, {product['price']}"
            else:
                label = barcode.data.decode()

            self.__mark_barcode(image, barcode, label)

    def __mark_face(self, image, face_location, label):
        """
        Marks the detected face in the input image with a rectangle and label.
        The face is marked with a colored rectangle and the label is displayed above the rectangle.
        """
        color = self.__name_to_color(label)

        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        cv2.rectangle(image, top_left, bottom_right, color, self.frame_thickness)

        text_placement = (face_location[3], face_location[0] - 6)
        cv2.putText(image, label, text_placement, cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size, (0, 0, 0), self.font_thickness + 3)
        cv2.putText(image, label, text_placement, cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size, (200, 200, 200), self.font_thickness)

    def __mark_barcode(self, image, barcode_location, label):
        """
        Marks the detected face in the input image with a rectangle and label.
        The face is marked with a colored rectangle and the label is displayed above the rectangle.
        """
        rectangle_color = [255, 0, 0]
        polylinies_color = [0, 255, 0]

        top_left = (barcode_location.rect.left, barcode_location.rect.top)
        bottom_right = (barcode_location.rect.left + barcode_location.rect.width,
                        barcode_location.rect.top + barcode_location.rect.height)

        cv2.rectangle(image, top_left, bottom_right, rectangle_color, self.frame_thickness)
        cv2.polylines(image, [np.array(barcode_location.polygon)], True, polylinies_color, self.frame_thickness)

        text_placement = (barcode_location.rect.left, barcode_location.rect.top - 6)
        cv2.putText(image, label, text_placement, cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size, (0, 0, 0), self.font_thickness + 3)
        cv2.putText(image, label, text_placement, cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size, (200, 200, 200), self.font_thickness)

    @staticmethod
    def __name_to_color(string: str) -> list[int]:
        """
        Take 3 first letters, tolower()
        lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
        :param string: String, which first 3 letters will be used for picking colour
        :return: RGB array of a colour
        """
        if string == "Customer":
            return [0, 255, 0]

        return [(ord(c.lower()) - 97) * 8 for c in string[:3]]

    @staticmethod
    def __darken_color(old_color: list[int], factor: float = 0.5) -> tuple[int, ...]:
        """
        Darken a given color by a factor.
        Factor should be between 0 and 1, where 0 is black and 1 is the original color.
        """
        return tuple(max(0, int(c * factor)) for c in old_color)
