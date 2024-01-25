import cv2
import numpy as np


class GUIModule:
    """
    A class to manage the GUI display for a robotic shopping assistant using OpenCV.

    Attributes:
    -----------
    video : cv2.VideoCapture
        The video capture object for camera access.
    frame_thickness : int
        The thickness of the frames for drawn objects.
    font_thickness : int
        The thickness of the font used in text.
    font_size : float
        The size of the font used in text.
    gui_colour : tuple
        The colour of the GUI elements.
    text_colour : tuple
        The colour of the text in the GUI.
    show_shopping_list : bool
        Flag to control the visibility of the shopping list in the GUI.

    Methods:
    --------
    get_image_frame(self)
        Captures an image frame from the camera and returns it along with any key pressed.

    display_detected_objects(self, image, detected_faces=None, detected_barcodes=None)
        Draws detected faces and barcodes on the image frame.

    toggle_shopping_list_visibility(self)
        Toggles the visibility of the shopping list on the GUI.

    render_gui(self, image, cart: list, total_cost)
        Renders the GUI overlay on the image frame.

    __mark_face(self, image, face_location, label)
        Internal method to mark a detected face on the image frame.

    __mark_barcode(self, image, barcode_location, label)
        Internal method to mark a detected barcode on the image frame.

    __name_to_color(string: str)
        Static method to generate a color based on a string.

    __darken_color(old_color: list[int], factor: float = 0.5)
        Static method to darken a given color.
    """
    def __init__(self, camera_width, camera_height, frame_thickness=2, font_thickness=2, font_size=0.6, gui_colour=(255, 255, 255)):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        self.frame_thickness = frame_thickness
        self.font_thickness = font_thickness
        self.font_size = font_size
        self.gui_colour = gui_colour
        self.text_colour = (0, 0, 0)
        self.show_shopping_list = False

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_image_frame(self):
        """
        Captures an image frame from the camera.
        """
        ret, image = self.video.read()
        key_pressed = cv2.waitKey(1) & 0xFF
        return image, key_pressed

    def display_detected_objects(self, image, detected_faces=None, detected_barcodes=None):
        """
        Draws rectangles and labels for detected faces and barcodes on the image frame.
        """
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

    def toggle_shopping_list_visibility(self):
        """
        Toggles the visibility of the shopping list on the GUI.
        """
        self.show_shopping_list = not self.show_shopping_list

    def render_gui(self, image, cart: list, total_cost):
        """
        Renders the GUI overlay on the image frame, including the shopping list and total cost.
        """
        gui = np.zeros_like(image, np.uint8)
        box_width = 200
        box_height = 50
        tc_box_x = image.shape[1] - box_width
        tc_box_y = image.shape[0] - box_height
        cv2.rectangle(gui, (tc_box_x, tc_box_y), (tc_box_x + box_width, tc_box_y + box_height), self.gui_colour, -1)

        rect_x = 0
        rect_y = 0

        # Shopping list
        if self.show_shopping_list:
            rect_height = image.shape[0]
            rect_width = image.shape[1] // 2
            cv2.rectangle(gui, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), self.gui_colour, -1)

        alpha = 0.5
        mask = gui.astype(bool)
        image[mask] = cv2.addWeighted(image, alpha, gui, 1 - alpha, 0)[mask]

        if self.show_shopping_list:
            element_x = 10
            element_y = 30
            for product in cart:
                cv2.putText(image, product['name'], (rect_x + element_x, rect_y + element_y),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.text_colour, self.font_thickness)
                element_y += 30

        # Total cart cost
        text = f'Total: {total_cost:.2f} [PLN]'
        text_x = tc_box_x + 10
        text_y = tc_box_y + 30
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size, self.text_colour, self.font_thickness)

        cv2.imshow("Robot's Sight", image)

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
                    self.font_size, self.text_colour, self.font_thickness + 3)
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
                    self.font_size, self.text_colour, self.font_thickness + 3)
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
