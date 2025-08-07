import zxingcpp
import numpy as np
import cv2
from pyzbar import pyzbar
from PIL import Image
from itertools import chain, combinations

def reduce_color(image: np.ndarray, color_divisor: int) -> np.ndarray:
    image_copy = image.copy()
    image_copy = color_divisor * ( image_copy // color_divisor ) + color_divisor // 2
    return image_copy

class UniqueColorIterator:
    ''' Iterates over all unique colors in an image, after dividing RGB values by an input divisor. '''
    def __init__(self, image: np.ndarray, color_divisor: int, threshold_pixel_average: int = 250):
        self._image = image
        self._color_divisor = color_divisor
        self._threshold_pixel_average = threshold_pixel_average

    def __filter_colors(self, image: np.array, colors: np.array): 
        remaining_colors = []
        for color in colors:
            # threshold on the specified color
            lower=np.array((color))
            upper=np.array((color))
            mask = cv2.inRange(image, lower, upper)

            single_color = image.copy()
            # change all non-specified color to white
            single_color[mask!=255] = (255, 255, 255)

            pixel_average = np.average(single_color)
            if pixel_average < self._threshold_pixel_average:
                remaining_colors.append(color)

        return remaining_colors

    def __iter__(self):
        # Do simple color reduction
        image_copy = reduce_color(self._image, self._color_divisor)

        # Get list of unique colors
        list_bgr_colors = np.unique(image_copy.reshape(-1, image_copy.shape[2]), axis=0)

        # Filter the list of unique colors, removing those colors with relatively few non-white pixels
        self._unique_colors = self.__filter_colors(image_copy, list_bgr_colors)
        self._i = 0

        return self
    
    def __next__(self):
        if self._i < len(self._unique_colors):
            color = self._unique_colors[self._i]
            self._i += 1
            return color
        else:
            raise StopIteration

class ColorSetIterator:
    ''' Iterates over all possible combinations of unique colors in an image, after dividing RGB values by an input divisor. '''
    def __init__(self, image: np.ndarray, color_divisor: int, threshold_pixel_average: int = 250):
        self._image = image
        self._color_divisor = color_divisor
        self._threshold_pixel_average = threshold_pixel_average

    def __powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def __iter__(self):
        # Materialize the list of unique colors
        unique_colors = [ color for color in UniqueColorIterator(self._image, self._color_divisor, self._threshold_pixel_average) ]
        
        # Construct the powerlist, skipping the empty set
        self._color_sets = [ s for s in self.__powerset(unique_colors) if len(s) > 0 ]
        self._i = 0

        return self
    
    def __next__(self):
        if self._i < len(self._color_sets):
            color_set = self._color_sets[self._i]
            self._i += 1
            return color_set
        else:
            raise StopIteration

class BarcodeReader():
    '''
    Encapsulates the barcode reading functionality.
    '''

    def __init__(self, skip_barcode_validation: bool = False):
        '''
        Initializes the BarcodeReader.
        
        :param skip_barcode_validation: If True, skips validation of barcodes.
        '''
        self.skip_barcode_validation = skip_barcode_validation

    def __is_valid_barcode(self, barcode: str) -> bool:
        if self.skip_barcode_validation:
            return True
        
        barcode_parts = barcode.split('-')
        n = len(barcode_parts)
        if n == 1:
            try:
                int_barcode = int(barcode_parts[0])
                return int(1E6) <= int_barcode and int_barcode < int(1E7)
            except ValueError:
                pass
        elif n == 3:
            try:
                int_lms = int(barcode_parts[1])
                int_barcode = int(barcode_parts[2])
                return barcode_parts[0] == 'LMS' and 1 <= int_lms and int_lms <= 9 and int(1E6) <= int_barcode and int_barcode < int(1E7)
            except ValueError:
                pass
        return False
    
    def read_file_path(self, file_path: str) -> str:
        # Check for barcode in file name
        file_path_parts = file_path.split(' ')
        barcode = file_path_parts[-1]
        if barcode is not None and self.__is_valid_barcode(barcode):
            return barcode
    
        return None

    def read_image(self, image: Image, validate: bool = True) -> str:
        # First attempt to read the barcode using zxing
        zxing_barcodes = zxingcpp.read_barcodes(image)
        if len(zxing_barcodes) > 0:
            barcode = zxing_barcodes[0].text
    
            if not validate or self.__is_valid_barcode(barcode):
                return barcode
    
        # Second attempt to read the barcode using pyzbar
        pyzbar_barcodes = pyzbar.decode(image)
        if len(pyzbar_barcodes) > 0:
            barcode = pyzbar_barcodes[0].data.decode('utf-8')
    
            if not validate or self.__is_valid_barcode(barcode):
                return barcode
            
        return None
    
    def read_rgb_array(self, image: np.ndarray) -> str:
        return self.read_image(self.rgb_array_to_image(image), False)
        
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
    
    def identify_colors(self, image: np.ndarray, color_divisor: int = 32, threshold_pixel_average: int = 250) -> list[np.ndarray]:
        return [ color for color in UniqueColorIterator(image, color_divisor, threshold_pixel_average) ]
    
    def identify_color_sets(self, image: np.ndarray, color_divisor: int = 32, threshold_pixel_average: int = 250) -> list[np.ndarray]:
        return [ color_set for color_set in ColorSetIterator(image, color_divisor, threshold_pixel_average) ]
    
    def reduce_and_read(self, image: np.ndarray, color_divisor: int, color_set: np.ndarray) -> tuple[str, np.ndarray]:
        # Apply color divisor
        reduced_image = reduce_color(image, color_divisor)

        transformed_image = None     

        for color in color_set:
            # threshold on the specified color
            lower=np.array((color))
            upper=np.array((color))
            mask = cv2.inRange(reduced_image, lower, upper)

            single_color = reduced_image.copy() 
            # change all non-specified color to white
            single_color[mask!=255] = (255, 255, 255)

            if transformed_image is None:
                transformed_image = single_color
            else:
                transformed_image = transformed_image + single_color

        barcode = self.read_rgb_array(transformed_image)
        return barcode, transformed_image
