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
# todo : clahe,
#       bilateral filter [?],
#       write bw images,
#       create pdf [with reducing image size],
#       heif convert
#       apply using default values


class ImageProcessor:

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_counter = 1

        self.color_space = {'color': cv2.IMREAD_COLOR,
                            'grayscale': cv2.IMREAD_GRAYSCALE,
                            }


    def image_processing_pipeline(self, **config):
        """Runs processes enabled in constructor."""

        external_config = config
        color_space = self.color_space[external_config['color_space']]

        # Create list of images for processing.
        os.chdir(self.input_dir)
        to_process: list[str] = [item for item in os.listdir(self.input_dir)
                                 if os.path.isfile(os.path.abspath(item))]

        # Apply selected processes to each image
        for item in to_process:
            # If os.chdir is placed out of loop only Open cv gets errors.
            os.chdir(self.input_dir)
            # Open image as Open cv object
            image_object: MatLike = cv2.imread(item, color_space)

            # Check if given functionality is enabled in config.
            if external_config['run_rotate_adjustment'][0]:
                # Store config params as variable.
                rotate_params = external_config['run_rotate_adjustment'][1]
                image_object = self.rotate_image(image_object,
                                                 rotation_angle=rotate_params[
                                               'rotation_angle']
                                                 )

            if external_config['reverse_colors'][0]:
            # No config params, since method does only color inversion.
                image_object = self.reverse_color(image_object)

            if (external_config['contrast_brightness'][0]
                    and not external_config['clahe'][0]):
                c_b_params = external_config['contrast_brightness'][1]
                image_object = self.contrast_brightness(image_object,
                                                        alpha=c_b_params[
                                                            'alpha'],
                                                        beta=c_b_params[
                                                            'beta']
                                                        )

            if external_config['sharpen_image'][0]:
                sharpen_params = external_config['sharpen_image'][1]
                image_object = self.sharpen_image(image_object,
                                                  kernel=sharpen_params[
                                                      'kernel'],
                                                  strength=sharpen_params[
                                                      'strength']
                                                  )

            if external_config['bilateral_filter'][0]:
                bilateral_params = external_config['bilateral_filter'][1]
                image_object = self.bilateral_filter(image_object,
                                                     d=bilateral_params['d'],
                                                     sigma_color=bilateral_params['sigma_color'],
                                                     sigma_space=bilateral_params['sigma_space']
                                                     )

            if external_config['denoise_image'][0]:
                denoise_params = external_config['denoise_image'][1]
                image_object = self.denoise_image(image_object,
                                                  color_space=color_space,
                                                  filter_strength=
                                                  denoise_params[
                                                      'filter_strength']
                                                  )

            if external_config['write_output_file'][0]:
                write_params = external_config['write_output_file'][1]
                self.write_output_file(image_object,
                                       output_dir=self.output_dir,
                                       file_extension=write_params[
                                           'file_extension'],
                                       quality=write_params['quality']
                                       )

        # Reset file counter after all files in dir have been processed.
        self.file_counter = 1

    @staticmethod
    def bilateral_filter(image_object: MatLike,
                         d: int = 9,
                         sigma_color: int = 75,
                         sigma_space:int = 75) -> MatLike:
        """
        Bilateral filter as alternative to other noise removal techniques.
        :param image_object: Open cv object, ie. image file converted to numpy
         array.
        :param d: Diameter of the pixel neighbourhood used for filtering;
         if d == 0 diameter is calculated based only on sigma_space.
          Default value == 9.
        :param sigma_color: Color deviation value. Higher value means
         higher tonal spread, ie. colors farther away from each other
          will be mixed.
          Default value == 75.
        :param sigma_space: Second parameter defining extent of pixel
         neighbourhood; higher value means that the further pixels will be
          mixed if their colors lie within sigma_color range.
          Default value == 75.
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """

        image = image_object
        image = cv2.bilateralFilter(image,
                                    d=d,
                                    sigmaColor=sigma_color,
                                    sigmaSpace=sigma_space)

        return image

    @staticmethod
    def contrast_brightness(image_object: MatLike,
                            alpha: int|float,
                            beta: int) -> MatLike:
        """
        Contrast and brightness adjustment, apply when not using clahe.
        :param image_object: Open cv object, ie. image file converted to numpy
         array.
        :param alpha: Contrast value. Value between 0 and 1 lowers the
         contrast, while value above 1 increases it.
        :param beta: Brightness value. Suggested value between -127 and 127.
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """
        image = image_object
        alpha_value = alpha
        beta_value = beta

        image = cv2.convertScaleAbs(image, alpha=alpha_value, beta=beta_value)

        return image

    @ staticmethod
    def denoise_image(image_object: MatLike,
                      color_space: int,
                      filter_strength: int) -> MatLike:
        """
        Apply denoising filter to an Open cv object. Apply to noised images
         or after using sharpening kernel.
        :param image_object: Open cv object, ie. image file converted to numpy
         array.
        :param color_space: Color space set for processed images; color or
         grayscale
        :param filter_strength: Higher value means better noise removal at
         the cost of removing image details and distorting colors (in color
         images).
         Recommended values:
            colors - 10
            grayscale - 30
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """
        image = image_object
        denoised_image = None
        # size in pixels of the template patch used to compute weights;
        # should be odd number; recommended value == 7
        template_window_size = 7
        # size in pixels of the window that is used to compute weighted
        # average for given pixel; should be odd number; affects
        # performance; recommended value == 21
        search_window_size = 21

        if color_space == 1:
            denoised_image = cv2.fastNlMeansDenoisingColored(
                src=image,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size,
                hColor=filter_strength
            )
        elif color_space == 0:
            denoised_image = cv2.fastNlMeansDenoising(
                src=image,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size,
                h=filter_strength
            )

        return denoised_image

    @staticmethod
    def reverse_color(image_object: MatLike) -> MatLike:
        """
        Reverse image colors. Useful for negative microforms or to enhance
         visibility of fading writing.
        :param image_object: Open cv object, ie. image file converted to numpy
         array.
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """
        image = image_object
        image = cv2.bitwise_not(image)

        return image

    @staticmethod
    def rotate_image(image_object: MatLike,
                     rotation_angle: int,
                     ) -> MatLike:
        """
        Rotate Open cv object by given value.
        :param image_object: Open cv object, ie. image file converted to numpy
         array.
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

    @staticmethod
    def sharpen_image(image_object: MatLike,
                      kernel: str,
                      strength: int = 0
                      ) -> MatLike:
        """
        Apply sharpen or unsharp masking on an image using kernels.
        It is possible to add more kernels in future if needed.
        :param image_object: Open cv object, ie. image file converted to numpy
         array.
        :param kernel: Kernel chosen from 'kernels' dict.
        :param strength: Multiplier for a kernel ranging from 1.00 (default)
         to 1.99.
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """
        filter_strength = strength

        # Check if strength value is between 0 and 99.
        if 0 > filter_strength or 100 < filter_strength:
            print(f'Strength value should be between 0 and 99, got '
                  f'{filter_strength} instead.\n'
                  'Applying default value.')
            filter_strength = 0

        # Standard sharpening kernel from Wikipedia.
        sharpening_kernel = np.multiply(float(f'1.{filter_strength}'),
                                        np.array([[0, -1, 0],
                                                [-1, 5, -1],
                                                [0, -1, 0]])
                                        )

        # Just like above, untested as of 16.04.25. exp 0.00425
        unsharp_masking_kernel = np.multiply(0.00390625 * float(
                                        f'1.{filter_strength}'),
                                    np.array([[1, 4, 6, 4, 1],
                                          [4, 16, 24, 16, 4],
                                          [6, 24, 46, 24, 6],
                                          [4, 16, 24, 16, 4],
                                           [1, 4, 6, 4, 1]])
                                             )

        # Dict of defined kernels to choose from.
        kernels = {'sharpen': sharpening_kernel,
                   'unsharp_mask': unsharp_masking_kernel,
                   }

        # Applying chosen kernel to image.
        apply_kernel = cv2.filter2D(image_object, -1, kernels[kernel])
        return apply_kernel

    #@ staticmethod
    #def apply_clahe(image: MatLike,
    #                           ):

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


config = {'color_space': 'color',
          'run_rotate_adjustment': [True, {'rotation_angle': 90}],
          'reverse_colors': [False],
          'write_output_file': [True,
                                {'file_extension': 'jpg',
                                 'quality': 60
                               }],
          'contrast_brightness': [False,
                                  {'alpha': 1.5,
                                  'beta': -50
                                   }],
          'clahe': [False],
          'sharpen_image': [True, {'kernel': 'sharpen',
                                   'strength': 0
                                           }],
          'bilateral_filter': [False,
                               {'d': 9,
                                'sigma_color': 75,
                                'sigma_space': 75
                                }],
          'denoise_image': [True, {'filter_strength': 10}]
        }


image_processor = ImageProcessor(input_dir=folder_path,
                                 output_dir=output_path)

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

