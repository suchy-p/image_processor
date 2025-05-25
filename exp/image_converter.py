import os

import cv2
import pillow_heif


class ImageConverter:
    def __init__(self, input_path: str,
                 output_file_extension: str,
                 output_file_name: str|None = None,
                 quality: int|None = None,
                 ):
        
        self.input_path = input_path
        self.output_file_extension = output_file_extension
        self.output_file_name = output_file_name
        self.quality = quality
        self.write_quality_param: list [int|None] = [None, None]

    def validate_inputs(self):
        extensions = ('jpg', 'jpeg', 'png')
        errors = []
        write_quality_param = self.write_quality_param

        # Validate file extension.
        if self.output_file_extension not in extensions:
            errors.append('Wrong extension. Choose jpg, jpeg or png.\n')

        if self.quality is not None:
            if self.output_file_extension is 'jpeg' or 'jpg':
                assert self.quality in range(0, 101), errors.append(
                    'Quality value should be an integer between 0 and 100.\n'
                )
                write_quality_param = [int(cv2.IMWRITE_JPEG_QUALITY),
                                       self.quality
                                       ]
            elif self.output_file_extension is 'png':
                assert self.quality in range(0, 10), errors.append(
                    'Compression value should be an integer between 0 and 9.\n'
                )
                write_quality_param = [int(cv2.IMWRITE_PNG_COMPRESSION),
                                       self.quality
                                       ]

        if len(errors) > 0:
            return errors
        else:
            self.write_quality_param = write_quality_param

    @staticmethod
    def rename(input_path:str,
            output_file_name:str):

        file_list = os.listdir(input_path)
        file_suffix = os.path.splitext(file_list[0])[1]

        new_file_name = f'{output_file_name}_'
        counter = 0

        os.chdir(input_path)
        for file in file_list:
            os.rename(file, f'{new_file_name.zfill(4), counter}.{file_suffix}')
            counter += 1

    @staticmethod
    def change_format():





def convert_heic(heic_path, output_file_extension, quality):
    output_path = os.path.join(heic_path, 'jpeg')
    counter = 1
    write_quality_param = [int(cv2.IMWRITE_JPEG_QUALITY),
                                    quality
                           ]

    # Change jpeg quality to png compression if output_file_extension == png.
    if output_file_extension == 'png':
        write_quality_param: list[int | None] = [
            int(cv2.IMWRITE_PNG_COMPRESSION),
            quality
        ]
        # Check if provided png compression factor is correct when not
        # using default value.
        if quality is not None:
            assert quality in range(0, 10), 'Compression value should ' \
                                            'be integer between 0 and 9.'

            # Check if provided jpg quality value is correct when not using
            # default value.
    if output_file_extension == 'jpg' and quality is not None:
        assert quality in range(0, 101), 'Quality value should be ' \
                                         'integer between 0 and 100.'

    file_list = os.listdir(heic_path)
    for f in file_list:
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