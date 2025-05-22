import zxingcpp
import numpy as np
import cv2
from pyzbar import pyzbar
from PIL import Image

class BarcodeReader():
    '''
    Encapsulates the barcode reading functionality.
    '''
    def __is_valid_barcode(self, barcode: str) -> bool:
        barcode_parts = barcode.split('-')
        n = len(barcode_parts)
        if n == 1:
            int_barcode = int(barcode_parts[0])
            return int(1E6) <= int_barcode and int_barcode < int(1E7)
        elif n == 3:
            int_lms = int(barcode_parts[1])
            int_barcode = int(barcode_parts[2])
            return barcode_parts[0] == 'LMS' and 1 <= int_lms and int_lms <= 9 and int(1E6) <= int_barcode and int_barcode < int(1E7)
        return False
    
    def read_file_path(self, file_path: str) -> str:
        # Check for barcode in file name
        file_path_parts = file_path.split(' ')
        barcode = file_path_parts[-1]
        if barcode is not None and self.__is_valid_barcode(barcode):
            return barcode
    
        return None

    def read_image(self, image: Image) -> str:
        # First attempt to read the barcode using zxing
        zxing_barcodes = zxingcpp.read_barcodes(image)
        if len(zxing_barcodes) > 0:
            barcode = zxing_barcodes[0].text

            if self.__is_valid_barcode(barcode):
                return zxing_barcodes[0].text

        # Second attempt to read the barcode using pyzbar
        pyzbar_barcodes = pyzbar.decode(image)
        if len(pyzbar_barcodes) > 0:
            barcode = pyzbar_barcodes[0].data.decode('utf-8')

            if self.__is_valid_barcode(barcode):
                return pyzbar_barcodes[0].data.decode('utf-8')
            
        return None
    
    def read_rgb_array(self, image: np.ndarray) -> str:
        return self.read_image(self.rgb_array_to_image(image))
        
    def image_to_rgb_array(self, image: Image) -> np.ndarray:
        '''
        Converts a PIL image to a numpy array in RGB format.
        '''
        # Convert the image to RGB format
        rgb_image = image.convert('RGB')
        # Convert the RGB image to a numpy array
        rgb_array = np.array(rgb_image)
        return rgb_array
    
    def rgb_array_to_image(self, rgb_array: np.ndarray) -> Image:
        return Image.fromarray(np.uint8(rgb_array))
    
    def increase_brightness(self, image: np.array, value: int = 30):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return image
    
    def increase_contrast(self, image: Image, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8,8)):
        lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))

        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img
    
    def resize_image(self, image: np.ndarray, target_size: tuple[int, int] = (440, 370)) -> np.ndarray:
        return cv2.resize(image, target_size)
    
    def crop_and_read(self, image: np.ndarray, a: int, b: int, c: int, d: int) -> tuple[str, np.ndarray]:
        cropped_image = image[c:d, a:b]
        barcode = self.read_rgb_array(cropped_image)
        return barcode, cropped_image
    
    def reduce_colors(self, image: np.ndarray, color_set: np.ndarray) -> np.ndarray:
        transformed_image = None     

        for color in color_set:
            # threshold on the specified color
            lower=np.array((color))
            upper=np.array((color))
            mask = cv2.inRange(image, lower, upper)

            single_color = image.copy() 
            # change all non-specified color to white
            single_color[mask!=255] = (255, 255, 255)

            if transformed_image is None:
                transformed_image = single_color
            else:
                transformed_image = transformed_image + single_color

        return transformed_image