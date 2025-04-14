import os

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import pillow_heif
from pypdf import PdfReader, PdfWriter
import img2pdf

heic_path: str = None
folder_path: str = ('C:\\Users\\Patryk\\Desktop\\Tygodnik '
                    'Rolniczo-Przemysłowy')
output_path: str = os.path.join(folder_path, 'opencv_')


def heif_convert(heic_path, jpeg_quality):
    save_file_path = os.path.join(heic_path, 'jpeg')
    counter = 1
    jpg_encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    print(heic_path)

    files = os.listdir(heic_path)
    for f in files:
        if f.endswith('.HEIC'):

            for i in os.listdir(heic_path):
                os.chdir(heic_path)
                heif_file = pillow_heif.open_heif(i,
                                                  convert_hdr_to_8bit = False,
                                                  bgr_mode=True,
                                                  )
                np_array = np.asarray(heif_file)

                # contrast limited adaptive histogram equalization; outside loop
                clip_limit: int = 5
                tile_grid_size: tuple[int, int] = (8, 8)
                clahe: cv2.CLAHE = cv2.createCLAHE(clip_limit, tile_grid_size)

                lab = cv2.cvtColor(np_array, cv2.COLOR_BGR2LAB)
                # splitting lab to lightness [0], green-red [1] and blue-yellow
                # [2] planes
                lab_planes = list(cv2.split(lab))
                # apply clahe to lightness
                lab_planes[0] = clahe.apply(lab_planes[0])
                # merging planes
                lab = cv2.merge(lab_planes)
                np_array = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                # resize 2/3 of original image
                np_array = cv2.resize(np_array, (0,0), fx=0.66, fy=0.66)

                # check existing output path
                if not os.path.isdir(save_file_path):
                    os.mkdir(save_file_path)

                os.chdir(save_file_path)

                cv2.imwrite(f'Image_{str(counter).zfill(3)}.jpg', np_array, jpg_encode_param)
                print(f'Image_{str(counter).zfill(3)}.jpg')
                counter += 1



# todo: add all subfunc args to grayscale_opencv, refactor for selective
#  subfunc usage passing and deafult subfunc params

def grayscale_opencv(input_folder_path: str,
                     file_extension: str,
                     rotate_angle: int = None,
                     jpeg_quality: int = 65,
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

        if item.endswith(('.jpg', '.jpeg')):

            # for grayscale and binary images operations
            # image: np.ndarray = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
            # for color file operations
            image: np.ndarray = cv2.imread(item, 0)

            # rotate right, left, flip vertical if rotate_angle argument is
            # provided
            if rotate_angle is not None:
                # rotate image as specified in rotate variable
                try:
                    image: np.ndarray = cv2.rotate(image,
                                                      rotate[rotate_angle],
                                                      )
                except KeyError:
                    print('Wprowadź wartość liczbę całkowitą oznaczającą '
                          'stopnie: 90, 180 lub 270')
                    break

            # contrast limited adaptive histogram equalization; outside loop
            clip_limit: int = 5
            tile_grid_size: tuple[int, int] = (8, 8)
            clahe: cv2.CLAHE = cv2.createCLAHE(clip_limit, tile_grid_size)

            # contrast (alpha) and brightness (beta) adjustments, optional;
            # variables outside loop;
            # use when not using clahe
            alpha: int|float = 0.1
            beta: int = 1
            image: np.ndarray = cv2.convertScaleAbs(image, alpha, beta)




            '''
            #clahe for color
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # splitting lab to lightness [0], green-red [1] and blue-yellow
            # [2] planes
            lab_planes = list(cv2.split(lab))
            # apply clahe to lightness
            lab_planes[0] = clahe.apply(lab_planes[0])
            # merging planes
            lab = cv2.merge(lab_planes)
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            '''

            # clahe for grayscale
            # image: np.ndarray = clahe.apply(image)

            # sharpening
            # sharpening kernel ; kerel outside loop
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

            # applying sharpening kernel as filter
            sharpened = cv2.filter2D(image, -1, kernel)
            image = sharpened

            # grayscale image denoising
            # source file
            source_file = image
            # size in pixels of the template patch used to compute weights;
            # should be odd number; recommended value == 7
            template_window_size = 7
            # size in pixels of the window that is used to compute weighted
            # average for given pixel; should be odd number; affects
            # performance; recommended value == 21
            search_window_size = 21
            # parameter regulating filter strength;
            # higher h ==> better noise removal ==> remove more details
            h = 30
            # sames as h but for color components; for most images hColor ==
            # 10 will be enough to remove noise but not distort colors
            h_color = 10

            image: np.ndarray = cv2.fastNlMeansDenoising(
                                    src=source_file,
                                    templateWindowSize=template_window_size,
                                    searchWindowSize=search_window_size,
                                    h=h
                                    )
            '''
            # color image denoising
            
            image: np.ndarray = cv2.fastNlMeansDenoisingColored(
                                    src=source_file,
                                    templateWindowSize=template_window_size,
                                    searchWindowSize=search_window_size,
                                    h=h,
                                    hColor=h_color
                                    )
            '''
            # bilateral filter applied as better for preserving edges
            # d: diameter of pixel neighborhood; if d == 0 ; vars outside loop
            # diameter is calculated based only on sigmaSpace
            d: int = 9
            # 2nd value - sigmaColor: color differences, higher value =
            # higher tonal spread
            sigma_color: int = 5
            # 3rd value - sigmaSpace: neighboring pixels
            sigma_space: int = 15
            image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

            # Reverse color, for negative images
            # grayscale_image: np.ndarray = cv2.bitwise_not(grayscale_image)

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
            constant: int = 40 #20 #10

            image: np.ndarray = cv2.adaptiveThreshold(
                                        src=image,
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
                '''
                cv2.imwrite(f'Image_{str(file_counter).zfill(3)}.'
                            f'{file_extension}',
                            image, png_encode_param
                            )
                '''
                # write jpg files with compression
                # better for grayscale images
                cv2.imwrite(f'Image_{str(file_counter).zfill(3)}.'
                            f'{file_extension}',
                           image,
                           jpg_encode_param
                           )
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

    if not os.path.isdir(output_path+'pdf'):
        os.mkdir(output_path+'pdf')
    os.chdir(im_files_path)

    try:
        # converting all images in directory to binary images
        '''
        images = [Image.open(i).convert('1') for i in os.listdir(
            im_files_path) if i.endswith(im_files_ext)]
        '''
        images = [i for i in os.listdir(im_files_path) if i.endswith(
            im_files_ext)]



        # create pdf form first image in dir, then appending rest of files
        for image in images:
            im = Image.open(image)
            im.save(
                pdf_file_name,'PDF', dpi=(10, 8 ))

        save_dir = output_path + 'pdf'
        single_pdfs = [p for p in os.listdir(save_dir)]
        os.chdir(save_dir)
        merger = PdfWriter()
        for pdf in single_pdfs:
            merger.append(PdfReader(pdf), 'rb')

        with open('out.pdf', 'wb') as file:
            merger.write(file)

        '''
        images[0].save(
            pdf_file_name,
            save_all=True,
            append_images=images[1:]
        )
        '''

        to_compress = PdfWriter(pdf_file_name)
        for page in to_compress.pages:
            page.compress_content_streams()

        with open('compressed.pdf', 'wb') as file:
            to_compress.write(file)

    except UnidentifiedImageError as e:
        print(f'Prawdopodobnie w folderze znajdują się pliki, które nie są '
              f'plikami graficznymi. Obsługiwane formaty to jpg i png. '
              f'\nKomunikat błędu: \n{e}')


def create_pdf_compressed(im_files_path: str,
                          im_files_ext: str,
                          save_file_name: str):

    pdf_file_name = f'{save_file_name}.pdf'
    save_dir = output_path + 'pdf'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    os.chdir(im_files_path)
    print (os.getcwd())

    #try:
    images = [os.path.abspath(i) for i in os.listdir(im_files_path)
              if i.endswith(im_files_ext)]

    for i in images:
        print (i)
    counter = 1

    os.chdir(save_dir)
    for image in images:

        im = Image.open(image)
        single_pdf = img2pdf.convert(im.filename)
        name = f'image_{str(counter).zfill(3)}.pdf'

        im.save(
            name, 'PDF')
        #with open (f'{save_dir}\\{name}', 'wb') as file:
        #    file.write(single_pdf)
        counter += 1
        print(name)
    
    #except error as e
    single_pdfs = [p for p in os.listdir(save_dir)]
    os.chdir(save_dir)

    merger = PdfWriter()
    for pdf in single_pdfs:
        merger.append(PdfReader(pdf), 'rb')

    with open ('out.pdf', 'wb') as file:
        merger.write(file)


        #with open('out.pdf', 'wb') as :
        #    merger.append(pdf)
    print ('files merged, going to compress')

    to_compress = PdfWriter('out.pdf')
    for page in to_compress.pages:
        page.compress_content_streams(level=9)

    with open ('compressed.pdf', 'wb') as file:
        to_compress.write(file)

    print('compression finished')










if __name__ == '__main__':
    # heif_convert(heic_path, jpeg_quality=65)

    grayscale_opencv(input_folder_path=folder_path,
                     jpeg_quality=65,
                     rotate_angle=None,
                     file_extension='jpg'
                     )
    '''
    create_pdf(im_files_path=output_path,
               im_files_ext='jpg',
               save_file_name='Out_file'
               )
   
    create_pdf_compressed(im_files_path=output_path,
               im_files_ext='jpg',
               save_file_name='Out_file'
               )
    '''