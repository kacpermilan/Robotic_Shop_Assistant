import cv2
import numpy as np


class VisualizationModule:
    def __init__(self, frame_thickness=2, font_thickness=2, font_size=0.6):
        self.frame_thickness = frame_thickness
        self.font_thickness = font_thickness
        self.font_size = font_size
        self.show_shopping_list = False

    def display_detected_objects(self, image, detected_faces=None, detected_barcodes=None):
        if detected_barcodes is None:
            detected_barcodes = []

        if detected_faces is None:
            detected_faces = []

        for face, label in detected_faces:
            self.__mark_face(image, face, label)

        for barcode, product in detected_barcodes:
            if product is not None:
                label = f"{product['name']}, {product['price']}"
            else:
                label = barcode.data.decode()

            self.__mark_barcode(image, barcode, label)

    def display_gui(self, image, total_cost: float):
        gui = np.zeros_like(image, np.uint8)

        box_width = 200
        box_height = 50
        box_x = image.shape[1] - box_width
        box_y = image.shape[0] - box_height
        cv2.rectangle(gui, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)

        alpha = 0.5
        mask = gui.astype(bool)
        image[mask] = cv2.addWeighted(image, alpha, gui, 1 - alpha, 0)[mask]

        text = f'Total: {total_cost:.2f} [PLN]'
        text_x = box_x + 10
        text_y = box_y + 30
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size, (0, 0, 0), self.font_thickness)

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
