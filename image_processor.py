import os

import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image
from pypdf import PdfWriter


class ImageProcessor:

    def __init__(self, settings: dict):
        self.input_dir = os.path.expanduser(settings["paths"]["input_dir"])
        self.output_dir = os.path.join(self.input_dir, settings["paths"][
                                           "output_dir"])
        self.file_counter = 1

        self.color_space = {
            "color": cv2.IMREAD_COLOR,
            "grayscale": cv2.IMREAD_GRAYSCALE,
        }

    @staticmethod
    def check_defaults_overwrite(settings: dict[str, str | int | float |None],
                                 default_params: dict[str, str | int | float
                                                        | tuple [int, int]
                                                           | None]
                                 ) -> dict[str, str | int | float]:
        """
        Check if settings.toml overwrites default parameters of given process,
         i.e. if passes not None value for any parameter.
        :param default_params: Dict of method default params.
        :param settings: User settings passed from settings.toml file.
        :return: Dict of items in config which values are not None.
        """
        user_params = settings
        set_params = dict()

        for param in user_params:
            # Ignore enabled = true at the beginning of given process'
            # params in settings.toml.
            if param == "enabled":
                continue
            # User's param replaces default value.
            if user_params[param] != "None":
                set_params[param] = user_params[param]
            else:
                set_params[param] = default_params[param]

        return set_params

    def adjust_brightness_and_contrast(self,
                                       image: MatLike,
                                       settings: dict[
                                           str, str | int | float | None],
                                       ) -> MatLike:
        """
        Contrast and brightness adjustment. You may want to apply it when not
         using clahe.
        :param image: Open cv image object.
        :param settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """
        image = image
        default_params = {"alpha": 1.0, "beta": 0}
        # Check for non-default user settings.
        passed_params = self.check_defaults_overwrite(settings,
                                                      default_params)

        image = cv2.convertScaleAbs(image, **passed_params)

        return image

    def bilateral_filter(self,
                         image: MatLike,
                         settings: dict[str, str | int | float | None],
                         ) -> MatLike:
        """
        Bilateral filter as alternative to other noise removal techniques.
        :param image: Open cv image object.
        :param settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """
        default_params = {"d": 9,"sigmaColor": 75, "sigmaSpace": 75}
        # Check for non-default user settings.
        passed_values = self.check_defaults_overwrite(settings,
                                                      default_params
                                                      )
        image = cv2.bilateralFilter(image,
                                    **passed_values
                                    )

        return image

    def clahe(self,
              image: MatLike,
              color_mode: int,
              settings: dict[str, str | int | float | None],
              ) -> MatLike:
        """
        Apply contrast limited adaptive histogram equalization for
         increased readability, especially for darkened areas of image.
         Suggested for writing black and white output images, but can
          process color images also.
        :param image: Open cv image object.
        :param color_mode: Color mode set for processed images; color or
         grayscale.
        :param settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """
        default_params = {"clipLimit": 40,"tileGridSize": (8,8)}
        # Check for non-default user settings.
        passed_params = self.check_defaults_overwrite(settings,
                                                      default_params,
                                                      )

        image = image
        clahe = cv2.createCLAHE(**passed_params)

        if color_mode == 0:
            image = clahe.apply(image)

        elif color_mode == 1:
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

    def denoise_image(self,
                      image: MatLike,
                      color_mode: int,
                      settings: dict[str, str | int | float | None],
                      ) -> MatLike:
        """
        Apply denoising filter to an Open cv object. Apply to noised images
         or after using sharpening kernel.
        :param image: Open cv image object.
        :param color_mode: Color mode set for processed images; color or
         grayscale.
        :param settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """
        image = image
        default_params = {"templateWindowSize": 7, "searchWindowSize": 21,}

        if color_mode == 0:
            # Remove filter strength value for color images and set default
            # filter strength value in settings.
            if "hColor" in settings:
                del settings["hColor"]
            default_params["h"] = 30
            # Check for non-default user settings.
            passed_params = self.check_defaults_overwrite(settings,
                                                          default_params
                                                          )
            image = cv2.fastNlMeansDenoising(src=image,
                                                      **passed_params
                                                      )
        elif color_mode == 1:
            # Remove filter strength value for grayscale images and set
            # default filter strength value in settings.
            if "h" in settings:
                del settings["h"]
            default_params["hColor"] = 10
            # Check for non-default user settings.
            passed_params = self.check_defaults_overwrite(settings,
                                                          default_params
                                                          )
            image = cv2.fastNlMeansDenoisingColored(src=image,
                                                             **passed_params
                                                             )

        return image

    @staticmethod
    def reverse_color(image: MatLike) -> MatLike:
        """
        Reverse image colors. Useful for negative microforms or to enhance
         visibility of fading writing.
        :param image: Open cv image object.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """
        image = image
        image = cv2.bitwise_not(image)

        return image

    @staticmethod
    def rotate_image(image: MatLike,
                     settings: dict[str, int],
                     ) -> MatLike:
        """
        Rotate Open cv object by given value.
        :param image: Open cv image object.
        :param settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """
        # Dict of supported rotation angles.
        rotate: dict[int, int] = {90: cv2.ROTATE_90_CLOCKWISE,
                                  180: cv2.ROTATE_180,
                                  270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                                  }

        # Apply image rotation.
        image: MatLike = image
        rotated_image = cv2.rotate(image, rotate[settings["angle"]],
                                                     "Invalid rotation value."
                                                     )

        return rotated_image

    def sharpen_image(self,
                      image: MatLike,
                      settings: dict[str, str | int | float | None],
                      ) -> MatLike:
        """
        Perform sharpen or unsharp masking on an image using kernels.
        It is possible to add more kernels in future if needed.
        :param image: Open cv image object.
        :param settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """
        image = image
        default_params = {"kernel": "sharpen", "filter_strength": 0.5}

        # Check if strength value is between 0 and 1.
        # for refactoring
        # if settings["filter_strength"] is "None":
        #     pass
        # else:
        #     print(settings.get("filter_strength"))
        #     if 0 > settings["filter_strength"] > 1:
        #         print(f"Strength value should be between 0 and 1, got "
        #               f"{settings["filter_strength"]} instead.\n"
        #               "Applying default value.")
        #         settings["filter_strength"] = default_params["filter_strength"]

        # Standard sharpening kernel from Wikipedia.
        sharpening_kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])

        # Gaussian blur, also from Wiki; normalized for 0 - 1 range.
        unsharp_masking_kernel = np.array([[1, 4, 6, 4, 1],
                                          [4, 16, 24, 16, 4],
                                          [6, 24, 46, 24, 6],
                                          [4, 16, 24, 16, 4],
                                          [1, 4, 6, 4, 1]])/255

        # Dict of defined kernels to choose from.
        kernels_map = {"sharpen": sharpening_kernel,
                   "unsharp_mask": unsharp_masking_kernel,
                   }
        # Check for non-default user settings.
        passed_params = self.check_defaults_overwrite(settings,
                                                      default_params)
        chosen_kernel = kernels_map[passed_params["kernel"]]
        # Applying chosen kernel to image.
        apply_kernel = cv2.filter2D(image, -1, chosen_kernel)
        # Blend original and modified image, filter strength serves
        # as weight.
        image = cv2.addWeighted(src1=image,
                                alpha=1-passed_params["filter_strength"],
                                src2=apply_kernel,
                                beta=passed_params["filter_strength"],
                                gamma=0)

        return image

    def thresholding(self,
                     image: MatLike,
                     color_mode: int,
                     settings: dict[str, str | int | float | None],
                     ) -> MatLike:
        """
         Create black and white images using adaptive thresholding.
        :param image: Open cv image object.
        :param color_mode: Color mode set for processed images; color or
         grayscale.
        :param settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """

        image = image
        default_params = {"adaptiveMethod": 0,
                          "maxValue": 255,
                          "blockSize": 199,
                          "C": 40,
                          "thresholdType": cv2.THRESH_BINARY
                          }
        passed_params = self.check_defaults_overwrite(settings,
                                                      default_params)

        # Check color space in config, change color space to grayscale if
        # needed.
        if color_mode == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.adaptiveThreshold(
            src=image,
            **passed_params)

        return image

    @staticmethod
    def write_pdf_file(  # images_path: str,
                       images_file_type: str = "jpg",
                       pdf_name: str = "Document",
                       pdf_compression: int = 0
                       ) -> None:
        """
        Creates PDF file from processed images.
        :param images_path: Path to directory containing image files.
        :param images_file_type: Specify image file type, so this func
         doesn't try to create PDF from non-image files that could be in
          images directory (e.g. previously created PDF file).
        :param pdf_name: Name of PDF file containing all images from
         specified directory.
        :param pdf_compression: Compression factor for PDF file:
         from 0 (no compression) to 9 (highest compression). Default value = 0.
        :return: None, writes PDF file in directory containing images.
        """
        output_file_name = f"{pdf_name}.pdf"
        # Path for temp single-image pdfs, deleted after merging into one file.
        counter = 1

        # Create list of images for pdf convertion.
        images = [os.path.abspath(image) for image in os.listdir(
            os.getcwd()) if image.endswith(f".{images_file_type}")]

        # Create temp single-page pdfs.
        print("Creating temp pdf files. They will be automatically deleted "
              "after everything is done.")
        for image in images:
            current_image = Image.open(image)
            name = f"_tempfile_{str(counter).zfill(3)}.pdf"
            current_image.save(name, "PDF")
            counter += 1

        # Create list of single-image pdfs for merging.
        print(os.listdir(os.getcwd()), "\n")
        single_image_pdfs = [os.path.abspath(pdf) for pdf in
                             os.listdir(os.getcwd())
                             if pdf.startswith("_tempfile_")]
        print(single_image_pdfs)

        merger = PdfWriter()

        print("Merging temp pdf files.")
        # Merge single-image pdfs into temp pdf file.
        for pdf in single_image_pdfs:
            merger.append(pdf, "rb")

        with open("_tempfile_merged.pdf", "wb") as file:
            merger.write(file)

        # Compress temp PDF file if pdf_compression > 0 and create
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
            source_file = os.path.join(os.getcwd(), "_tempfile_merged.pdf")
            destination_path = os.path.join(os.getcwd(), output_file_name)

            if os.path.isfile(destination_path):
                os.remove(destination_path)
            os.rename(source_file, destination_path)

        # Delete temp files.
        for file in os.listdir(os.getcwd()):
            if file.startswith("_tempfile"):
                os.remove(file)

        print("Finished.")

    def write_processed_image(self,
                              image_object: MatLike,
                              settings,
                              output_dir: str,
                              file_extension: str = "jpg",
                              quality: int | None = None
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
        default_params = {"file_extension": "jpg",
                          "quality": 85}
        passed_params = self.check_defaults_overwrite(settings, default_params)
        output_dir = self.output_dir

        image: MatLike = image_object
        file_name = (f"Image_{str(self.file_counter).zfill(4)}."
                     f"{passed_params["file_extension"]}")
        quality_param: list[int | dict[str, int]] = [int(
            cv2.IMWRITE_JPEG_QUALITY), int(passed_params["quality"])
                                                 ]

        # Change jpeg quality to png compression if file_extension == png.
        if passed_params["file_extension"] == "png":
            quality_param: list[int | dict[str, int]] = [
                int(cv2.IMWRITE_PNG_COMPRESSION),
                int(passed_params["quality"])
            ]
            # Check if provided png compression factor is correct when not
            # using default value.
        #     if quality is not None:
        #         assert quality in range(0, 10), "Compression value should " \
        #                                         "be integer between 0 and 9."
        #
        # # Check if provided jpg quality value is correct when not using
        # # default value.
        # if file_extension == "jpg" and quality is not None:
        #     assert quality in range(0, 101), "Quality value should be " \
        #                                      "integer between 0 and 100."

        # Check for existing output directory, then change working dir.
        if not os.path.isdir(output_dir):
            print("isdir false")
            print(output_dir)
            os.mkdir(output_dir)
        os.chdir(output_dir)

        # Write Open cv object as image file.
        try:
            cv2.imwrite(file_name,
                        image,
                        quality_param
                        )
            print(f"{file_name} file created.")
            self.file_counter += 1
        # In case of typo in provided file_extension.
        except cv2.error as e:
            print(f"Probably invalid file extension. Choose jpg or png. \n "
                  f"Error message:\n {e}")
