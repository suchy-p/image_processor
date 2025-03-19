import os

import cv2
from PIL import Image, UnidentifiedImageError

folder_path = 'C:\\Users\\YaTeż\\Desktop\\Rola\\1 Rola Ossolineum — kopia\\'
output_path: str = os.path.join(folder_path, 'opencv_')


# todo: add all subfunc args to grayscale_opencv, refactor for selective
#  subfunc usage passing and deafult subfunc params
# todo: sharpen image for reverse colors
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

        if item.endswith('jpg'):

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
            '''
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
            '''
            grayscale_image: numpy.ndarray = clahe.apply(image)

            # grayscale image denoising
            # source file
            source_file = grayscale_image
            # size in pixels of the template patch used to compute weights;
            # should be odd number; recommended value == 7
            template_window_size = 7
            # uzupełnić
            search_window_size = 21
            # uzupełnić
            h = 7
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
            constant: int = 20
            bw_image: numpy.ndarray = cv2.adaptiveThreshold(
                                        src=grayscale_image,
                                        maxValue=max_value,
                                        adaptiveMethod=adaptive_method[0],
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

    grayscale_opencv(input_folder_path=folder_path,
                     jpeg_quality=50,
                     # rotate_angle=90,
                     file_extension='png'
                     )

    create_pdf(im_files_path=output_path,
               im_files_ext='png',
               save_file_name='Out_file'
               )
