import os
from contextlib import chdir

import cv2
import numpy as np
from PIL import Image
import pillow_heif
from pypdf import PdfWriter

from image_processor import ImageProcessor


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

    def image_converting_pipeline(self, **config_params: dict[
                                                         str:str|int|None]):
        config = config_params
        checker = ImageProcessor.checker

        validate: list|None = self.validate_inputs()

        if validate is not None:
            return validate

        if config['change_format'][0]:
            params = checker(config['change_format'][1])
            self.change_format(**params)

        if config['rename'][0]:
            params = checker(config['rename'][1])
            self.rename(**params)

        if config['write_pdf_file'][0]:
            params = checker(config['write_pdf_file'][1])
            self.write_pdf_file(**params)

        return None

    @staticmethod
    def change_format(file_list: list[str],
                      output_file_suffix: str,
                      output_file_path: str,
                      write_quality_param: list[int | None],
                      counter=0,
                      output_file_name: str = 'Image_',
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
                                              convert_hdr_to_8bit=False,
                                              bgr_mode=True,
                                              )
                image = np.asarray(image)

            else:
                image = cv2.imread(file)

            cv2.imwrite(name, image, quality)
            chosen_counter += 1

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
    def write_pdf_file(images_path: str,
                       images_file_type: str,
                       #output_path: str,
                       pdf_file_name: str,
                       pdf_file_compression: int = 0
                       ) -> None:
        """
        Creates pdf file from processed images.
        :param images_path: Path to directory containing image files.
        :param images_file_type: Specify image file type, so this func
         doesn't try to create pdf from non-image files that could be in
          images directory (e.g. previously created pdf file).
        :param output_path:
        :param pdf_file_name: Name of pdf file containing all images from
         specified directory.
        :param pdf_file_compression: Compression factor for pdf file:
         from 0 (no compression) to 9 (highest compression). Default value = 0.
        :return: None, writes pdf file in directory containing images.
        """
        output_file_name = f'{pdf_file_name}.pdf'
        pdf_compression = pdf_file_compression
        # Path for temp single-image pdfs, deleted after merging into one file.
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

        with open('_tempfile_merged.pdf', 'wb') as file:
            merger.write(file)

        # Compress temp pdf file if pdf_file_compression > 0 and create
        # output file. For me it doesn't seem to work at all, leaving it
        # here just in case.
        if pdf_compression > 0:
            pdf_to_compress = PdfWriter('_tempfile_merged.pdf')
            print('Compressing pdf file.')
            for page in pdf_to_compress.pages:
                page.compress_content_streams(level=pdf_compression)

            with open(output_file_name, 'wb') as file:
                pdf_to_compress.write(file)

        else:
            # If compression value == 0, copy temp pdf file as output file.
            source_file = os.path.join(images_path, '_tempfile_merged.pdf')
            destination_path = os.path.join(images_path, output_file_name)

            if os.path.isfile(destination_path):
                os.remove(destination_path)
            os.rename(source_file, destination_path)

        # Delete temp files.
        for file in os.listdir(images_path):
            if file.startswith('_tempfile'):
                os.remove(file)

        print('Finished.')






conf = {

}

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

