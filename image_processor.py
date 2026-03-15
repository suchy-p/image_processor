import os

import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image
from pypdf import PdfWriter


# args for class instance
folder_path: str = ('C:\\Users\\Patryk\\Desktop\\stress')
output_path: str = os.path.join(folder_path, 'opencv_')
rotate_angle: int = 90
# todo :
#       create pdf [with reducing image size],
#       refactor above


class ImageProcessor:

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_counter = 1

        self.color_space = {'color': cv2.IMREAD_COLOR,
                            'grayscale': cv2.IMREAD_GRAYSCALE,
                            }

    def image_processing_pipeline(self, **config: dict)->None :
        """
        Runs processes enabled in configuration dict, passes params to
         selected processes.
        :param config: Dict containing configuration options.
        :return: None, applies selected processes to image files.
        """

        external_config = config
        color_space = self.color_space[external_config['color_space']]

        # Create list of images for processing.
        os.chdir(self.input_dir)
        to_process: list[str] = [item for item in os.listdir(os.getcwd())
                                 if os.path.isfile(item)]
        print(to_process)

        # Apply selected processes to each image
        for item in to_process:
            # If os.chdir is placed out of loop only Open cv gets errors.
            os.chdir(self.input_dir)
            # Open image as Open cv object
            image_object: MatLike = cv2.imread(item, color_space)

            # Check if given functionality is enabled in config.
            if external_config['run_rotate_adjustment'][0]:
                # Store config params as variable.
                params = external_config['run_rotate_adjustment'][1]
                image_object = self.rotate_image(image_object,
                                                 rotation_angle=params[
                                               'rotation_angle']
                                                 )

            if external_config['reverse_colors'][0]:
            # No config params, since method does only color inversion.
                image_object = self.reverse_color(image_object)

            if external_config['clahe'][0]:
                params = self.checker(external_config['clahe'][1])
                image_object = self.clahe(image_object,
                                          color_space=color_space,
                                          **params
                                          )

            if external_config['contrast_brightness'][0]:
                params = self.checker(external_config[
                                          'contrast_brightness'][1])
                image_object = self.contrast_brightness(image_object,
                                                        **params
                                                        )

            if external_config['sharpen_image'][0]:
                params = self.checker(external_config['sharpen_image'][1])
                image_object = self.sharpen_image(image_object,
                                                  ** params
                                                  )

            if external_config['bilateral_filter'][0]:
                params = self.checker(external_config['bilateral_filter'][1])
                image_object = self.bilateral_filter(image_object,
                                                     **params
                                                     )

            if external_config['denoise_image'][0]:
                params = self.checker(external_config['denoise_image'][1])
                image_object = self.denoise_image(image_object,
                                                  color_space=color_space,
                                                  **params
                                                  )

            if external_config['black_and_white'][0]:
                params = self.checker(external_config['black_and_white'][1])
                image_object = self.black_and_white(image_object,
                                                    **params
                                                    )

            if external_config['write_processed_image'][0]:
                params = self.checker(external_config['write_processed_image'][1])
                self.write_processed_image(image_object,
                                       output_dir=self.output_dir,
                                       **params
                                       )

        if external_config['write_pdf_file'][0]:
            params = self.checker(external_config['write_pdf_file'][1])
            self.write_pdf_file(**params)

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
    def black_and_white(image_object: MatLike,
                        method: str = 'mean',
                        max_value: int = 255,
                        block_size: int = 199,
                        constant: int = 40) -> MatLike:
        """
         Create black and white images using adaptive thresholding.
        :param image_object: Open cv object, ie. image file converted to numpy
         array.
        :param method: Choose between thresholding methods: mean or gaussian.
        :param max_value: Max value assigned to pixel.
        :param block_size: Size of pixel neighborhood used to calculate
         threshold value. Should be odd number.
        :param constant: Value subtracted from the mean or weighted (gaussian
          thresholding) sum of neighbouring pixels
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """

        image = image_object

        # Check color space in config, change color space to grayscale if
        # needed.
        if config['color_space'] == 'color':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        adaptive_method = {'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
                           'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C}

        image = cv2.adaptiveThreshold(
            src=image,
            maxValue=max_value,
            adaptiveMethod=adaptive_method[method],
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=constant)

        return image

    @staticmethod
    def checker(params: dict[str, str | int | float | None]) -> dict:
        """
        Check if config overwrites default parameters of given process,
         ie. if passes not None value for any parameter.
        :param params: Parameters from config dictionary.
        :return: Dict of items in config which values are not None.
        """
        check_params = params
        not_none_values = dict()

        for param in check_params:
            if check_params[param] is not None:
                not_none_values[param] = check_params[param]

        return not_none_values

    @staticmethod
    def clahe(image_object: MatLike,
              color_space: int,
              clip_limit: int = 40,
              tile_grid_size: tuple[int, int] = (8, 8)) -> MatLike:
        """
        Apply contrast limited adaptive histogram equalization for
         increased readability, especially for darkened areas of image.
         Suggested for writing black and white output images, but can
          process color images also.
        :param image_object: Open cv object, ie. image file converted to numpy
         array.
        :param color_space: Color space set for processed images; color or
         grayscale.
        :param clip_limit: Threshold for contrast limiting. Default value: 40.
        :param tile_grid_size: Sets row and column size of tile used to
         divide image for applying clahe. Default value: 8 rows, 8 columns.
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """

        image = image_object
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)

        if color_space == 0:
            image = clahe.apply(image)

        elif color_space == 1:
            # Convert image to lab color space.
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # Split lab to lightness [0], green-red [1] and blue-yellow
            # [2] planes.
            lab_planes = list(cv2.split(lab))
            # Apply clahe to lightness plane.
            lab_planes[0] = clahe.apply(lab_planes[0])
            # Merge all planes.
            lab = cv2.merge(lab_planes)
            # Convert lab to bgr color space.
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return image

    @staticmethod
    def contrast_brightness(image_object: MatLike,
                            alpha: int|float = 1,
                            beta: int = 0) -> MatLike:
        """
        Contrast and brightness adjustment. You may want to apply it when not
         using clahe.
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
                      filter_strength: int = 10) -> MatLike:
        """
        Apply denoising filter to an Open cv object. Apply to noised images
         or after using sharpening kernel.
        :param image_object: Open cv object, ie. image file converted to numpy
         array.
        :param color_space: Color space set for processed images; color or
         grayscale.
        :param filter_strength: Higher value means better noise removal at
         the cost of removing image details and distorting colors (in color
         images); default value: 10.
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

        if color_space == 0:
            denoised_image = cv2.fastNlMeansDenoising(
                src=image,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size,
                h=filter_strength
            )
        elif color_space == 1:
            denoised_image = cv2.fastNlMeansDenoisingColored(
                src=image,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size,
                hColor=filter_strength
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
                      kernel: str = 'sharpen',
                      strength: int = 0
                      ) -> MatLike:
        """
        Apply sharpen or unsharp masking on an image using kernels.
        It is possible to add more kernels in future if needed.
        :param image_object: Open cv object, ie. image file converted to numpy
         array.
        :param kernel: Kernel chosen from 'kernels' dict. Default value:
         sharpen.
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

    @ staticmethod
    def write_pdf_file(images_path: str,
                       images_file_type: str,
                       pdf_file_name: str,
                       pdf_file_compression: int = 0
                       ) -> None:
        """
        Creates pdf file from processed images.
        :param images_path: Path to directory containing image files.
        :param images_file_type: Specify image file type, so this func
         doesn't try to create pdf from non-image files that could be in
          images directory (e.g. previously created pdf file).
        :param pdf_file_name: Name of pdf file containing all images from
         specified directory.
        :param pdf_file_compression: Compression factor for pdf file:
         from 0 (no compression) to 9 (highest compression). Default value = 0.
        :return: None, writes pdf file in directory containing images.
        """
        output_file_name = f'{pdf_file_name}.pdf'
        pdf_compression = pdf_file_compression
        # Path for temp single-image pdfs, deleted after merging into one file.
        temp_dir_path = '_temp'
        counter = 1

        # Create list of images for pdf convertion.
        images = [os.path.abspath(image) for image in os.listdir(
            images_path) if image.endswith(f'.{images_file_type}')]

        # Create temp single-page pdfs.
        print('Creating temp pdf files. They will be automatically deleted '
              'after everything is done.')
        for image in images:
            current_image = Image.open(image)
            name = f'_tempfile_{str(counter).zfill(3)}.pdf'
            current_image.save(name, 'PDF')
            counter += 1

        # Create list of single-image pdfs for merging.
        single_image_pdfs = [os.path.abspath(pdf) for pdf in
                             os.listdir(images_path)
                             if pdf.startswith('_tempfile_')]

        merger = PdfWriter()

        print('Merging temp pdf files.')
        # Merge single-image pdfs into temp pdf file.
        for pdf in single_image_pdfs:
            merger.append(pdf, 'rb')

        with open ('_tempfile_merged.pdf', 'wb') as file:
            merger.write(file)

        # Compress temp pdf file if pdf_file_compression > 0 and create
        # output file. For me it doesn't seem to work at all, leaving it
        # here just in case.
        if pdf_compression > 0:
            pdf_to_compress = PdfWriter('_tempfile_merged.pdf')
            print('Compressing pdf file.')
            for page in pdf_to_compress.pages:
                page.compress_content_streams(level=pdf_compression)

            with open (output_file_name, 'wb') as file:
                pdf_to_compress.write(file)

        else:
            # If compression value == 0, copy temp pdf file as output file.
            source_file = os.path.join(output_path, '_tempfile_merged.pdf')
            destination_path = os.path.join(output_path, output_file_name)

            if os.path.isfile(destination_path):
                os.remove(destination_path)
            os.rename(source_file, destination_path)

        # Delete temp files.
        for file in os.listdir(output_path):
            if file.startswith('_tempfile'):
                os.remove(file)

        print('Finished.')

    def write_processed_image(self,
                          image_object: MatLike,
                          output_dir: str,
                          file_extension: str = 'jpg',
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
        :return: None, writes image file of chosen file type.
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

image_processor = ImageProcessor(input_dir=folder_path,
                                 output_dir=output_path
                                 )



