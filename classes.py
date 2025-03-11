import os

import cv2

from PIL import Image, ImageOps

folder_path = 'C:\\Users\\YaTeż\\Desktop\\Rola\\Rola 27.02.25 — kopia\\'


class ImageReadEnhance:
    """
    Create grayscale and binary images for enhanced readability and better
    ocr results using Pillow and OpenCV.
    For best results create binary images from grayscale.
    """

    def __init__(self):
        self.path = str()
        self.input_folder_path = folder_path
        self.output_folder_path = str()

        self.file_counter = 1
        self.file_extension = 'jpg'
        self.file_name = (f'Image_{str(self.file_counter).zfill(4)}.'
                     f'{self.file_extension}')

    def opencv_grayscale(self, path):
        self.path = path
        self.output_folder_path = os.path.join(self.path, 'grayscale_opencv')

        for item in os.listdir(self.path):
            # W/o changing working dir an error occurs.
            # Same with writing output file.
            os.chdir(self.path)

            image = cv2.imread(item)

            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if not os.path.isdir(self.output_folder_path):
                os.mkdir(self.output_folder_path)
            os.chdir(self.output_folder_path)

            cv2.imwrite(self.file_name, grayscale_image)
            print(f'{self.file_name} saved.')
            self.file_counter += 1

    def pillow_grayscale(self, path):
        self.path = path
        self.output_folder_path = os.path.join(self.path,'grayscale_pillow')

        for item in os.listdir(self.path):
            image = Image.open(os.path.join(self.input_folder_path, item))
            grayscale_image = ImageOps.grayscale(image).rotate(270)

            if not os.path.isdir(self.output_folder_path):
                os.mkdir(self.output_folder_path)

            grayscale_image.save(
                os.path.join(self.output_folder_path, self.file_name)
            )
            print(f'{self.file_name} saved.')
            self.file_counter += 1

processor = ImageReadEnhance()

# processor.pillow_grayscale(folder_path)
processor.opencv_grayscale(folder_path)

if __name__ == '__classes__':
    classes()