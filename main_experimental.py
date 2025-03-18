import os
import warnings

import PIL.Image
import cv2
'''
todo:
- multi image pdf
- png compression  
'''
import numpy as np

from PIL import Image, ImageOps, ImageEnhance

folder_path = 'C:\\Users\\YaTeż\\Desktop\\Rola\\Rola 27.02.25 — kopia\\'



# warnings.filterwarnings('ignore')
def grayscale_opencv():
    file_counter = 1
    output_path = os.path.join(folder_path, 'grayscale_opencv')

    for item in os.listdir(folder_path):
        os.chdir(folder_path)

        if item.endswith('jpg'):


            image = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image = cv2.equalizeHist(image)
            # bilateral filter applied as better for preserving edges
            image = cv2.bilateralFilter(image, 9, 5, 5)
            #image = cv2.medianBlur(image, 3)
            #image = cv2.GaussianBlur(image, (5, 5), 0)

            #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            #hsv_image = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            #h, s, v = cv2.split(hsv_image)



            blurred = cv2.bilateralFilter(image, 9, 75, 75) # adjust values

            clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,
                                                                8)) #clipLimit=40,
            #lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            #lab[:, :, 0] = clahe.apply(lab[:, :, 0])

            # alpha - contrast; beta - brightness
            #alpha = 1.5
            #beta = -120

            #image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            #img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            #v = clahe.apply(v)
            #hsv_image = cv2.merge([h, s, v])
            #hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

            #grayscale_image1 = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
            #grayscale_image = cv2.addWeighted(im, 0.2, grayscale_image1,
            #                                  0.8, 20)
            # grayscale_image = cv2.addWeighted(img_clahe, 0.7, img, 0.3, -10)
            grayscale_image = clahe.apply(image)
            #bw_image = cv2.adaptiveThreshold(grayscale_image, 255,
            #                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                         cv2.THRESH_BINARY, 199, 50)
            bw_image = cv2.adaptiveThreshold(grayscale_image, 255,
                                             cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 199, 50)
            # Odwrócenie koloru dla negatywów
            # bw_image = cv2.bitwise_not(bw_image)

            '''
            cv2.imshow('image', blurred)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),60]
            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            os.chdir(output_path)

            cv2.imwrite(f'Image_{str(file_counter).zfill(3)}.png',
                        bw_image)
            #cv2.imwrite('Image_bw.jpg', bw_image, encode_param)
            #cv2.imwrite('Image_bw_experimental.png', bw_image)

            print(f'Saved: Image_{str(file_counter).zfill(3)}.png.')
            file_counter += 1



def grayscale_pillow():
    file_counter = 1
    output_path = os.path.join(folder_path, 'grayscale_pillow')
    pil_data = None
    for item in os.listdir(folder_path):

        if item.endswith('jpg'):

            image = Image.open(os.path.join(folder_path, item)).rotate(270,
                                                                       resample=1,
                                                                       expand=True)


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

            output = ImageOps.grayscale(image)

            output.save(os.path.join(output_path, f'Image_{str(file_counter).zfill(3)}.jpg'))

            print(f'Image_{str(file_counter).zfill(3)}.jpg saved.')
            file_counter += 1
            file_path = os.path.join(output_path, f'Image_'
                                                f'{str(file_counter).zfill(3)}.jpg')
            pil_data = PIL.Image.open(file_path).convert('RGB')
            break
    #return pil_data



# grayscale_pillow()
grayscale_opencv()

