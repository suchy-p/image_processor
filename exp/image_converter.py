import os
from contextlib import chdir
from os import getcwd
from shutil import copyfile

import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image
import pillow_heif
from pypdf import PdfWriter

from image_processor import ImageProcessor


class ImageConverter:
    def __init__(self, input_path: str):

        self.input_path: str = input_path
        self.output_file_suffix: str = "jpg"
        self.write_quality_param: list[int | None] = [None, None]

        self.file_list: list[str] = os.listdir(self.input_path)
        self.output_file_path: str = os.path.join(self.input_path,
                                                  "converted_files")
        print(self.input_path)

    def image_converting_pipeline(self,
                                  **config_params: dict[str,
                                                        str | int | None]) -> None:
        """
        Runs processes enabled in configuration dict, passes params to
         selected processes.
        :param config_params: Dict containing configuration options.
        :return: None, applies selected processes to image files.
        """
        config = config_params
        checker = ImageProcessor.check_defaults_overwrite

        # Validate user inputs.
        validate: list | None = self.validate_inputs(config)

        if validate is not None:
            print(validate)
            return None

        else:
            if not os.path.isdir(self.output_file_path):
                os.mkdir(self.output_file_path)
            chdir(self.input_path)
            # If config[process_name][0] is set to True check if user
            # overwrites default values.
            if config["change_format"][0]:
                params = checker(config["change_format"][1])
                self.change_format(**params)

            if config["rename"][0]:
                params = checker(config["rename"][1])
                self.rename(**params)

            if config["write_pdf_file"][0]:
                params = checker(config["write_pdf_file"][1])
                self.write_pdf_file(**params)

        return None

    def change_format(self,
                      quality: int,
                      counter: int = 0,
                      output_file_name: str = "Image_",
                      output_file_suffix: str = None,
                      ) -> None:
        """
        Change format of image files in selected directory.
        :param quality: Desired quality/compression value. Passed to
         validator, then used to set write_quality_param in constructor; it
          stays here, because it's the logical place.
        :param counter: Starting file number, default: 0.
        :param output_file_name: Name of new files, default: Image_.
        :param output_file_suffix: Desired file format, passed by instance.
        :return: This method doesn't return anything.
        """

        # Check for any stray PDFs, temp files which weren't deleted because
        # of prior errors.
        file_list = [file for file in self.file_list if not file.endswith(
            "pdf")]
        output_file_suffix = output_file_suffix
        current_counter = counter

        os.chdir(self.input_path)

        for file in file_list:
            name = (f"{output_file_name}{str(current_counter).zfill(4)}."
                    f"{output_file_suffix}")

            # Check for heic files.
            if os.path.splitext(file)[1].lower() == ".heic":
                image = pillow_heif.open_heif(file,
                                              convert_hdr_to_8bit=False,
                                              bgr_mode=True,
                                              )
                # Convert image var to np array, so cv2 can read it.
                image: MatLike = np.array(image)

            else:
                image: MatLike = cv2.imread(file)

            # Write file using provided params.
            cv2.imwrite(os.path.join(self.output_file_path, name), image,
                        self.write_quality_param)
            current_counter += 1

    def rename(self,
               output_file_name: str,
               counter: int = 0) -> None:
        """
        Rename chosen files. Keeps file format by getting suffix of first file.
        :param output_file_name: Desired file name.
        :param counter: Staring file number, default: 0.
        :return: This method doesn't return anything.
        """
        # Get suffix if first file to keep consistent file format.
        file_suffix = os.path.splitext(self.file_list[0])[1]
        new_file_name = f"{output_file_name}_"
        use_counter = counter
        print(file_suffix)

        os.chdir(self.input_path)
        files = [file for file in os.listdir(os.getcwd()) if
                 os.path.isfile(file)]
        print(files)
        for file in files:
            dest_path = os.path.join(os.getcwd(), "converted_files")
            dest_filename = (f"{new_file_name}{str(use_counter).zfill(4)}"
                             f"{file_suffix}")
            copyfile(file, os.path.join(dest_path, dest_filename))
            use_counter += 1

    def validate_inputs(self, config: dict) -> list | None:
        """
        Validate files selected for conversion, chosen output file format
         and quality / compression value.
        :return: If passed: None, changes self.write_quality_param; if
         failed: exceptions list.
        """
        extensions = ("jpg", "jpeg", "png", "tiff", "heic")
        exceptions = []
        quality = config["change_format"][1]["quality"]

        config = config
        desired_format = (config["change_format"][1]["output_file_suffix"])

        # Validate list of files for conversion: check file type.
        for file in self.file_list:
            if os.path.splitext(file)[1].lower() not in extensions:
                self.file_list.remove(file)
        assert len(self.file_list) > 0, exceptions.append(
            "There are no files of supported types in source directory."
        )

        # Validate chosen output file extension in extensions var; tiff and
        # HEIC excluded.
        if desired_format not in extensions[:3:]:
            exceptions.append("Wrong extension. Choose jpg, jpeg or png.\n")

        # Validate chosen file quality value if default value is overwritten;
        # change self.write_quality_params for chosen quality/compression
        # value.
        if quality is not None:
            if desired_format == "jpeg" or desired_format == "jpg":
                assert quality in range(0, 101), exceptions.append(
                    "Quality value should be an integer between 0 and 100.\n"
                )
                self.write_quality_param = [int(cv2.IMWRITE_JPEG_QUALITY),
                                            quality
                                            ]
            elif self.output_file_suffix == "png":
                assert quality in range(0, 10), exceptions.append(
                    "Compression value should be an integer between 0 and 9.\n"
                )
                self.write_quality_param = [int(cv2.IMWRITE_PNG_COMPRESSION),
                                            quality
                                            ]

        # Return exceptions if any has occurred.
        if len(exceptions) > 0:
            return exceptions
        else:
            return None

    @staticmethod
    def write_pdf_file(images_path: str,
                       pdf_file_name: str,
                       pdf_file_compression: int = 0
                       ) -> None:
        """
        Creates PDF file from processed images.
        :param images_path: Path to directory containing image files.
         doesn't try to create PDF from non-image files that could be in
          images directory (e.g. previously created PDF file).
        :param pdf_file_name: Name of PDF file containing all images from
         specified directory.
        :param pdf_file_compression: Compression factor for PDF file:
         from 0 (no compression) to 9 (highest compression). Default value = 0.
        :return: None, writes PDF file in directory containing images.
        """

        output_file_name = f"{pdf_file_name}.pdf"
        pdf_compression = pdf_file_compression
        counter = 0

        os.chdir(images_path)

        # Create new list of images for PDF convertion. Just in case files
        # were renamed or file format was changed in the same run.
        suffixes = ("jpg", "jpeg", "png")
        images = [image for image in os.listdir(getcwd()) if image.endswith(
            suffixes)]

        # Create temp single-page pdfs.
        print("Creating temp pdf files. They will be automatically deleted "
              "after everything is done.")
        for image in images:
            current_image = Image.open(image)
            name = f"_tempfile_{str(counter).zfill(3)}.pdf"
            current_image.save(name, "PDF")
            counter += 1

        # Create list of single-image pdfs for merging.
        single_image_pdfs = [pdf for pdf in os.listdir(images_path)
                             if pdf.startswith("_tempfile_")]

        merger = PdfWriter()

        print("Merging temp pdf files.")
        # Merge single-image pdfs into temp pdf file.
        for pdf in single_image_pdfs:
            merger.append(pdf, "rb")
        # Write merged pdfs.
        with open("_tempfile_merged.pdf", "wb") as file:
            merger.write(file)

        # Compress temp PDF file if pdf_file_compression > 0 and create
        # output file. For me, it doesn't seem to work at all, leaving it
        # here just in case.
        if pdf_compression > 0:
            pdf_to_compress = PdfWriter("_tempfile_merged.pdf")
            print("Compressing pdf file.")
            for page in pdf_to_compress.pages:
                page.compress_content_streams(level=pdf_compression)

            with open(output_file_name, "wb") as file:
                pdf_to_compress.write(file)

        else:
            # If compression value == 0, copy temp pdf file as output file.
            source_file = os.path.join(images_path, "_tempfile_merged.pdf")
            destination_path = os.path.join(images_path, output_file_name)

            # Check for preexisting file.
            if os.path.isfile(destination_path):
                os.remove(destination_path)

            os.rename(source_file, destination_path)

        # Delete temp files.
        for file in os.listdir(images_path):
            if file.startswith("_tempfile"):
                os.remove(file)

        print("Finished.")


conf = {"change_format": [False,
                          {"counter": None,
                           "output_file_name": None,
                           "output_file_suffix": "jpg",
                           "quality": 80
                           }],
        "rename": [False,
                   {"counter": None,
                    "output_file_name": "IMG",
                    }],
        "write_pdf_file": [False,
                           {"images_path":
                            "C:\\Users\\Patryk\\Desktop\\"
                            "Leon Beczek 1939–1943",
                            "pdf_file_name": "Document"
                            }]

        }

path_to_files = "C:\\Users\\Patryk\\Desktop\\Leon Beczek 1939–1943"

if __name__ == "__main__":
    im_converter = ImageConverter(input_path=path_to_files)
    im_converter.image_converting_pipeline(**conf)
