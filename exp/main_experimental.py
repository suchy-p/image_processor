import os
from typing import IO, BinaryIO

import cv2
from cv2.typing import MatLike
import mypy
import numpy as np
from PIL import Image, UnidentifiedImageError
from tomlkit.items import Array

# args for class instance
folder_path: str = ('C:\\Users\\Patryk\\Desktop\\Tygodnik '
                    'Rolniczo-Przemysłowy')
output_path: str = os.path.join(folder_path, 'opencv_')
rotate_angle: int = 90


class ImageProcessor:

    def __init__(self, input_dir, output_dir, rotation_angle, **kwargs):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_counter = 1
        self.rotation_angle = rotation_angle

        self.clahe = {'enabled': True,
                      'clip_limit': 5,
                      'tile_grid_size': (8, 8)
                      }
        self.contrast_brightness = {'enabled': False,
                                    'alpha': 1.5,
                                    'beta': -120
                                    }
        self.sharpen = {'enabled': True,
                        'value': 0
                        }, # add value to array
        self.bilateral_filter = {'enabled': False,
                                 'd': 9,
                                 'sigma_color': 5,
                                 'sigma_space': 5
                                 }
        self.black_white_threshold = {'enabled': True,
                                      'max_value': 255,
                                      'adaptive_method': 1,
                                      'block_size': 199,
                                      ' constant': 10
                                      }
        self.output_images = {'extension': '.png',
                              'quality': 5
                              }
                                # default bw png files,
                                # grayscale jpegs available on demand :
                                # can't be both on, add check

    def image_processing_pipeline(self, **config):
        """Runs processes enabled in constructor."""

        external_config = config
        # os.chdir(self.input_dir)
        # Create list of images for processing.
        to_process: list[str] = [item for item in os.listdir(self.input_dir)
                                 if os.path.isfile(os.path.abspath(item))]
        for item in os.listdir(self.input_dir):
            print(os.path.abspath(item), os.path.isfile(os.path.abspath(
                item)))

        #to_process: list[str] = os.listdir(self.input_dir)

        print(to_process)
        # Apply selected processes to each image
        for item in to_process:
            # If os.chdir is placed out of loop Open cv gets errors.
            os.chdir(self.input_dir)
            # Open image as Open cv object
            image_object: MatLike = cv2.imread(item)

            if self.rotation_angle:
                image_object = self.run_rotate_adjustment(image_object,
                                           rotation_angle=external_config[
                                               'rotation_angle']
                                                          )

            #if self.contrast_brightness['enabled']:
            #    self.run_contrast_brightness_adjustment

            #if self.clahe['enabled']:
            #    self.run_clahe_adjustment
            self.write_output_file(image_object,
                                   output_dir=external_config['output_dir'],
                                   file_extension=external_config[
                                       'file_extension'],
                                   quality=external_config['quality'])

        # Reset file counter after all files in dir have been processed.
        self.file_counter = 1


    def run_rotate_adjustment(self,
                              image_object: MatLike,
                              rotation_angle: int,
                              ) -> MatLike:
        """
        Rotate Open cv object by given value.
        :param image_object: Open cv object to rotate, ie. image file
         converted to numpy array.
        :param rotation_angle: Angle for rotating object clockwise by 90
         degrees steps: 90, 180 or 270 degrees.
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """
        # Dict of supported rotation angles.
        rotate: dict[int, int] = {90: cv2.ROTATE_90_CLOCKWISE,
                                  180: cv2.ROTATE_180,
                                  270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                                  }

        # Apply image rotation.
        image: MatLike = image_object
        rotated_image = cv2.rotate(image, rotate.get(rotation_angle,
                                                     'Invalid rotation value.'
                                                     )
                                   )

        return rotated_image

    def write_output_file(self,
                          image_object: MatLike,
                          output_dir: str,
                          file_extension: str,
                          quality: int|None = None
                          ) -> None:
        """
        Write Open cv object as image file of type chosen by user
        (suggested formats: jpg or png).
        :param image_object: Open cv object for writing as file.
        :param output_dir: Path for writing file, class parameter.
        :param file_extension: Image file format extension, suggested jpeg
         for color and png for black and white images. Must be passed by user.
        :param quality: Quality of jpg file in range from 0 to 100 or
         compression of png file in range form 0 to 9;
         if None Open cv applies default values: 95 for jpg, 3 for png.
        :return: Image file of chosen file type.
        """
        image_to_write: MatLike = image_object
        file_name = f'Image_{str(self.file_counter).zfill(4)}.{file_extension}'
        write_quality_param: list[int|None] = [int(cv2.IMWRITE_JPEG_QUALITY),
                                               quality
                                               ]

        # Change jpeg quality to png compression if file_extension == png.
        if file_extension == 'png':
            write_quality_param: list[int | None] = [
                int(cv2.IMWRITE_PNG_COMPRESSION),
                quality
            ]
            # Check if provided png compression factor is correct when not
            # using default value.
            if quality is not None:
                assert quality in range (0, 10), 'Compression value should '\
                                                'be integer between 0 and 9.'

        # Check if provided jpg quality value is correct when not using
        # default value.
        if file_extension == 'jpg' and quality is not None:
            assert quality in range (0, 101), 'Quality value should be '\
                                               'integer between 0 and 100.'

        # Check for existing output directory, then change working dir.
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        os.chdir(output_path)

        # Write Open cv object as image file.
        try:
            cv2.imwrite(file_name,
                        image_to_write,
                        write_quality_param
                        )
            print (f'{file_name} file created.')
            self.file_counter += 1
        # In case of typo in provided file_extension.
        except cv2.error as e:
            print(f'Probably invalid file extension. Choose jpg or png. \n '
                  f'Error message:\n {e}')

config = {'input_dir': 'C:\\Users\\Patryk\\Desktop\\Tygodnik Rolniczo'
                         '-Przemysłowy',
          'output_dir': str(os.path.join(folder_path, 'opencv_')),
          'rotation_angle': 90,
          'file_extension': 'jpg',
          'quality': 60
}



image_processor = ImageProcessor(**config)

image_processor.image_processing_pipeline(**config)




'''
# todo: add all subfunc args to grayscale_opencv, refactor for selective
#  subfunc usage passing and deafult subfunc params
# todo: sharpen image for reverse colors
# todo: add unsharp mask?
def grayscale_opencv(input_folder_path: str,
                     file_extension: str,
                     rotate_angle: int = None,
                     jpeg_quality: int = 85,
                     png_compression: int = 5,
                     ) -> None:
    """
    Create grayscale or binary image files for improved readability using
    Open CV 2.
    Useful for darkened documents photos or microforms photos with
    non-linear lights.
    :param input_folder_path: path to images for processing
    :param file_extension:  extension of output files, jpeg suggested for
    grayscale, png for binary images
    :param rotate_angle: angle of clockwise image rotation during processing
    in degrees: 90, 180, 270
    :param jpeg_quality: quality of output jpeg file from 1 to 100; default
    value: 85
    :param png_compression: png file compression, from 1 to 10; default
    value: 5
    :return: None | writes processed files in folder created in working
    directory
    """

    file_counter: int = 1
    # file_extension: str = 'jpg'
    rotate: dict[int, int] = {90: cv2.ROTATE_90_CLOCKWISE,
                              180: cv2.ROTATE_180,
                              270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                              }

    # params for output file compression
    jpg_encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    png_encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression]

    for item in os.listdir(folder_path):
        os.chdir(folder_path)

        if item.endswith(('.jpg', '.png')):

            image: numpy.ndarray = cv2.imread(item, cv2.IMREAD_GRAYSCALE)

            # rotate right, left, flip vertical if rotate_angle argument is
            # provided
            if rotate_angle is not None:
                # rotate image as specified in rotate variable
                try:
                    image: numpy.ndarray = cv2.rotate(image,
                                                      rotate[rotate_angle],
                                                      )
                except KeyError:
                    print('Wprowadź wartość liczbę całkowitą oznaczającą '
                          'stopnie: 90, 180 lub 270')
                    break

            # contrast limited adaptive histogram equalization
            clip_limit: int = 5
            tile_grid_size: tuple[int, int] = (8, 8)
            clahe: cv2.CLAHE = cv2.createCLAHE(clip_limit, tile_grid_size)

            # contrast (alpha) and brightness (beta) adjustments, optional
            # alpha: int|float = 1.5
            # beta: int = -120
            # image: numpy.ndarray = cv2.convertScaleAbs(image, alpha, beta)

            # sharpening
            # sharpening kernel
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

            # applying sharpening kernel as filter
            sharpened = cv2.filter2D(image, -1, kernel)
            grayscale_image = sharpened

            # bilateral filter applied as better for preserving edges
            # d: diameter of pixel neighborhood; if d == 0
            # diameter is calculated based only on sigmaSpace
            d: int = 9
            # 2nd value - sigmaColor: color differences, higher value =
            # higher tonal spread
            sigma_color: int = 5
            # 3rd value - sigmaSpace: neighboring pixels
            sigma_space: int = 5
            image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

            grayscale_image: numpy.ndarray = clahe.apply(image)

            # grayscale image denoising
            # source file
            source_file = grayscale_image
            # size in pixels of the template patch used to compute weights;
            # should be odd number; recommended value == 7
            template_window_size = 7
            # size in pixels of the window that is used to compute weighted
            # average for given pixel; should be odd number; affects
            # performance; recommended value == 21
            search_window_size = 21
            # uzupełnić
            h = 10
            
            grayscale_image: numpy.ndarray = cv2.fastNlMeansDenoising(
                                    src=source_file,
                                    templateWindowSize=template_window_size,
                                    searchWindowSize=search_window_size,
                                    h=h
                                    )
            
            # Reverse color, for negative images
            grayscale_image: numpy.ndarray = cv2.bitwise_not(grayscale_image)

            # Create black and white image using adaptive threshold.
            # max value assigned to pixel
            max_value: int = 255
            # adaptive thresholding method, index 0 == mean or 1 == gaussian
            adaptive_method = [cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C]
            # size of pixel neighborhood used to calculate threshold value
            block_size: int = 199
            # value subtracted from the mean or weighted (gaussian
            # thresholding) sum of neighbouring pixels
            constant: int = 10
            bw_image: numpy.ndarray = cv2.adaptiveThreshold(
                                        src=grayscale_image,
                                        maxValue=max_value,
                                        adaptiveMethod=adaptive_method[1],
                                        thresholdType=cv2.THRESH_BINARY,
                                        blockSize=block_size,
                                        C=constant)


            # check existing output path
            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            os.chdir(output_path)

            try:
                # write png files with compression
                # better for bw images
                cv2.imwrite(f'Image_{str(file_counter).zfill(3)}.'
                            f'{file_extension}',
                            bw_image, png_encode_param
                            )

                # write jpg files with compression
                # better for grayscale images
                # cv2.imwrite(f'Image_{str(file_counter).zfill(3)}.{
                # file_extension}',
                #            grayscale_image,
                #            jpg_encode_param
                #            )
            except cv2.error as e:
                print(f'Prawdopodobnie niewłaściwy format pliku. '
                      f'Wybierz jpeg lub png. \n Treść błędu:\n {e}')
                break

            print(f'Saved: Image_{str(file_counter).zfill(3)}.'
                  f'{file_extension}.')
            file_counter += 1


def create_pdf(im_files_path: str, im_files_ext: str, save_file_name: str):
    """
    Creates pdf file from image files in selected directory using PIL.
    :param im_files_path: path to folder containing image files
    :param im_files_ext: extension of image files, preferably png or jpeg
    :param save_file_name: name of created pdf file
    :return: None, creates pdf file in im_files_path dir
    """

    pdf_file_name = f'{save_file_name}.pdf'
    os.chdir(im_files_path)

    try:
        # converting all images in directory to binary images
        images = [Image.open(i).convert('1') for i in os.listdir(
            im_files_path) if i.endswith(im_files_ext)]
        # create pdf form first image in dir, then appending rest of files
        images[0].save(
            pdf_file_name,
            save_all=True,
            append_images=images[1:]
        )
    except UnidentifiedImageError as e:
        print(f'Prawdopodobnie w folderze znajdują się pliki, które nie są '
              f'plikami graficznymi. Obsługiwane formaty to jpg i png. '
              f'\nKomunikat błędu: \n{e}')


if __name__ == '__main__':
'''

