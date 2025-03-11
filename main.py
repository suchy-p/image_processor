import os
import warnings
import cv2

from PIL import Image, ImageOps, ImageEnhance

folder_path = 'C:\\Users\\YaTeż\\Desktop\\Rola\\Rola 27.02.25 — kopia\\'



# warnings.filterwarnings('ignore')
def grayscale_opencv():
    file_counter = 1
    output_path = os.path.join(folder_path, 'grayscale_opencv')
    for item in os.listdir(folder_path):
        os.chdir(folder_path)

        if item.endswith('jpg'):
            image = cv2.imread(item)

            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            grayscale_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
            # experimental code below
            #(row, col) = rotated_image.shape[0:2]

            #for i in range(row):
            #    for j in range(col):
            #        rotated_image[i, j] = sum(rotated_image[i, j]) * 0.33

            if not os.path.isdir(output_path):
                os.mkdir(output_path)
            os.chdir(output_path)

            cv2.imwrite(f'Image_{str(file_counter).zfill(3)}.jpg', grayscale_image)

            print(f'Saved: Image_{str(file_counter).zfill(3)}.jpg.')
            file_counter += 1

def grayscale_pillow():
    file_counter = 1
    output_path = os.path.join(folder_path, 'grayscale_pillow')
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
           # output.convert('1', dither=None)

            output.save(os.path.join(output_path, f'Image_{str(file_counter).zfill(3)}.jpg'))

            print(f'Image_{str(file_counter).zfill(3)}.jpg saved.')
            file_counter += 1


grayscale_pillow()

if __name__ == '__main__':
    main()