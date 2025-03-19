import os

import cv2
from PIL import Image, ImageOps, ImageEnhance
'''
todo:
- multi image pdf
'''


folder_path: str = 'C:\\Users\\YaTeż\\Desktop\\Rola\\Rola 27.02.25 — kopia\\'
output_path: str = os.path.join(folder_path, 'opencv_')

def grayscale_opencv(input_folder_path: str,
                     output_folder_path: str,
                     file_extension: str='jpeg',
                     rotate_angle: int=None,
                     jpeg_quality: int=85,
                     png_compression: int=5,
                     ) -> None:

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
            '''
            source_file = grayscale_image
            # size in pixels of the template patch used to compute weights;
            # should be odd number; recommended value == 7
            template_window_size = 7
            grayscale_image: numpy.ndarray = cv2.fastNlMeansDenoising(
                                    src=source_file,
                                    templateWindowSize=template_window_size,
                                    searchWindowSize=21,
                                    h=7
                                    )
            '''
            # Reverse color, for negative images
            # grayscale_image: numpy.ndarray = cv2.bitwise_not(grayscale_image)

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


def create_pdf(im_output_path:str):
    os.chdir(im_output_path)
    images = [Image.open(i).convert('1') for i in os.listdir(im_output_path)]

    images[0].save(
        'Out_file.pdf',
        save_all=True,
        append_images=images[1:]
    )



def grayscale_pillow():
    file_counter = 1
    output_path = os.path.join(folder_path, 'grayscale_pillow')

    for item in os.listdir(folder_path):

        if item.endswith('jpg'):
            image = Image.open(os.path.join(folder_path, item))
            image = image.rotate(270, resample=1, expand=True)

            grayscale_image = ImageEnhance.Color(image).enhance(-1.5)
            brightness_image = ImageEnhance.Brightness(
                grayscale_image).enhance(1.2)

            contrast_image = ImageEnhance.Contrast(
                brightness_image).enhance(0.8)

            sharpen_image = ImageEnhance.Sharpness(contrast_image).enhance(2)

            image_data = sharpen_image.getdata()
            lst = []
            for i in image_data:
                lst.append(i[0]*0.2125+i[1]*0.7174+i[2]*0.0721)

            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            output = Image.new('L', sharpen_image.size, )
            output.putdata(lst)

            output.convert('1', dither=None)

            # output = ImageOps.grayscale(image)

            output.save(os.path.join(output_path,
                                     f'Image_{str(file_counter).zfill(3)}'
                                     f'.jpg'), quality=70)

            print(f'Image_{str(file_counter).zfill(3)}.jpg saved.')
            file_counter += 1


if __name__ == '__main_experimental__':
     grayscale_pillow()
'''
grayscale_opencv(input_folder_path=folder_path,
                 output_folder_path=output_path,
                 jpeg_quality=50,
                 rotate_angle=90,
                 file_extension='png'
                 )
'''

    create_pdf(im_output_path=output_path)
