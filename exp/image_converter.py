import os
from contextlib import chdir

import cv2
import numpy as np
import pillow_heif


class ImageConverter:
    def __init__(self, input_path: str,
                 output_file_suffix: str,
                 output_file_name: str|None = None,
                 quality: int|None = None,
                 ):
        
        self.input_path = input_path
        self.output_file_suffix = output_file_suffix
        self.output_file_name = output_file_name
        self.quality = quality
        self.write_quality_param: list [int|None] = [None, None]

        self.file_list = os.listdir(self.input_path)
        self.output_file_path = os.path.join(self.input_path,
                                             'converted_files')

    def validate_inputs(self):
        """
        Validate files selected for conversion, chosen output file format
         and quality / compression value.
        :return: If passed: None, changes self.write_quality_param; if
         failed: exceptions list.
        """
        extensions = ('jpg', 'jpeg', 'png', 'tiff', 'heic')
        exceptions = []
        write_quality_param = self.write_quality_param

        # Validate list of files for conversion: check file type.
        for file in self.file_list:
            if os.path.splitext(file)[1].lower() not in extensions:
                self.file_list.remove(file)
        assert len(self.file_list) > 0, exceptions.append(
            'There are no files of supported types in source directory.'
        )
            
        # Validate chosen output file extension; tiff and heic excluded.
        if self.output_file_suffix not in extensions[-3::]:
            exceptions.append('Wrong extension. Choose jpg, jpeg or png.\n')
        
        # Validate chosen file quality value if not default.
        if self.quality is not None:
            if self.output_file_suffix is 'jpeg' or 'jpg':
                assert self.quality in range(0, 101), exceptions.append(
                    'Quality value should be an integer between 0 and 100.\n'
                )
                write_quality_param = [int(cv2.IMWRITE_JPEG_QUALITY),
                                       self.quality
                                       ]
            elif self.output_file_suffix is 'png':
                assert self.quality in range(0, 10), exceptions.append(
                    'Compression value should be an integer between 0 and 9.\n'
                )
                write_quality_param = [int(cv2.IMWRITE_PNG_COMPRESSION),
                                       self.quality
                                       ]
        
        # Return exceptions if any has occurred, else change
        # self.write_quality_params for chosen quality / compression value.
        if len(exceptions) > 0:
            return exceptions
        else:
            self.write_quality_param = write_quality_param
            return None

    @staticmethod
    def rename(input_path:str,
            output_file_name:str,
            file_list: list[str],
            counter = 0):
        """
        Rename chosen files.
        :param input_path: Path to selected files.
        :param output_file_name: Desired file name.
        :param file_list: List of validated files, passed by instance.
        :param counter: Staring file number, default: 0.
        :return: Nothing, staticmethod.
        """

        passed_file_list = file_list
        file_suffix = os.path.splitext(passed_file_list[0])[1]

        new_file_name = f'{output_file_name}_'
        use_counter = counter

        os.chdir(input_path)
        for file in passed_file_list:
            os.rename(file, f'{new_file_name.zfill(4), use_counter}'
                            f'.{file_suffix}')
            use_counter += 1

    @staticmethod
    def change_format(file_list:list[str],
                      output_file_suffix:str,
                      output_file_path:str,
                      write_quality_param:list[int|None],
                      counter = 0,
                      output_file_name:str = 'Image_',
                      ):
        """
         Change format of image files in selected directory.
        :param file_list: List of validated files, passed by instance.
        :param output_file_suffix: Desired file format, passed by instance.
        :param output_file_path: Folder created in images directory
         containing new files, passed by instance.
        :param write_quality_param: Quality / compression param, passed by
         instance.
        :param counter: Starting file number, default: 0.
        :param output_file_name: Name of new files, default: Image_.
        :return: Nothing, staticmethod.
        """
        passed_file_list = file_list
        suffix = output_file_suffix
        passed_output_dir = output_file_path
        quality = write_quality_param
        chosen_counter = counter

        if not os.path.isdir(passed_output_dir):
            os.mkdir(passed_output_dir)
        chdir(passed_output_dir)

        for file in passed_file_list:
            name = (f'{output_file_name}{str(chosen_counter).zfill(4)}.'
                    f'{suffix}')
            # Check for heic format.
            if os.path.splitext(file)[1].lower() == 'heic':
                image = pillow_heif.open_heif(file,
                                              convert_hdr_to_8bit = False,
                                              bgr_mode=True,
                                              )
                image = np.asarray(image)

            else:
                image = cv2.imread(file)

            cv2.imwrite(name, image, quality)
            chosen_counter += 1



'''
    def convert_heic(heic_path, output_file_suffix, quality):
        output_path = os.path.join(heic_path, 'jpeg')
        counter = 1
        write_quality_param = [int(cv2.IMWRITE_JPEG_QUALITY),
                                        quality
                               ]

        # Change jpeg quality to png compression if output_file_suffix == png.
        if output_file_suffix == 'png':
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
        if output_file_suffix == 'jpg' and quality is not None:
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
'''

