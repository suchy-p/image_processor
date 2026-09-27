import os
import sys

import cv2
from cv2.typing import MatLike
import numpy as np


class ImageProcessor:

    def __init__(self, settings: dict):
        self.input_dir = os.path.expanduser(settings["paths"]["input_dir"])
        self.output_dir = os.path.join(self.input_dir, settings["paths"]["output_dir"])
        # Remove file counter after moving file writing to separate module.
        # Also reminded in ImageProcessingPipeline.
        self.file_counter = 1

    @staticmethod
    def check_defaults_overwrite(user_settings: dict[str, str | int | float | None],
                                 default_settings: dict[str, str | int | float
                                                             | tuple [int, int]
                                                             | None]
                                 ) -> dict[str, str | int | float | tuple[int, int]]:
        """
        Checks if settings.toml overwrites default parameters of given process,
         i.e. if passes not None value for any parameter.

        :param default_settings: Dict of method default params.
        :param user_settings: User settings passed from settings.toml file.
        :return: Dict of items in config which values are not None.
        """

        output_settings = dict()

        for setting in user_settings:
            # Ignore enabled = true at the beginning of given process'
            # params in settings.toml.
            if setting == "enabled":
                continue
            # User's param replaces default value.
            if user_settings[setting] != "None":
                output_settings[setting] = user_settings[setting]
            else:
                output_settings[setting] = default_settings[setting]

        return output_settings

    def adjust_brightness_and_contrast(self,
                                       image: MatLike,
                                       user_settings: dict[
                                           str, str | int | float | None],
                                       ) -> MatLike:
        """
        Contrast and brightness adjustment.

        :param image: Open cv image object.
        :param user_settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """

        default_settings = {"alpha": 1.0, "beta": 0}
        # Check for non-default user settings.
        checked_settings = self.check_defaults_overwrite(user_settings,
                                                      default_settings)

        image = cv2.convertScaleAbs(image, **checked_settings)

        return image

    def bilateral_filter(self,
                         image: MatLike,
                         user_settings: dict[str, str | int | float | None],
                         ) -> MatLike:
        """
        Bilateral filter as alternative to other noise removal techniques.

        :param image: Open cv image object.
        :param user_settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """

        default_settings = {"d": 0,"sigmaColor": 75, "sigmaSpace": 75}
        # Check for non-default user settings.
        checked_settings = self.check_defaults_overwrite(user_settings,
                                                         default_settings
                                                         )

        image = cv2.bilateralFilter(image,
                                    **checked_settings
                                    )

        return image

    def clahe(self,
              image: MatLike,
              color_mode: int,
              user_settings: dict[str, str | int | float | None],
              ) -> MatLike:
        """
        Apply Contrast Limited Adaptive Histogram Equalization for
        increased readability, especially for darkened areas of an image.
        Suggested for writing black and white output images, but can
        also process color images.

        :param image: Open cv image object.
        :param color_mode: Color mode set for processed images; color or
         grayscale.
        :param user_settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """

        default_settings = {"clipLimit": 40.0,"tileGridSize": (8,8)}
        # Check for non-default user settings.
        checked_settings = self.check_defaults_overwrite(user_settings,
                                                      default_settings,
                                                      )

        clahe = cv2.createCLAHE(**checked_settings)

        if color_mode == 0:
            image = clahe.apply(image)

        elif color_mode == 1:
            # Convert image to LAB color space.
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # Split LAB to lightness [0], green-red [1] and blue-yellow
            # [2] planes.
            lab_planes = list(cv2.split(lab))
            # Apply CLAHE to lightness plane.
            lab_planes[0] = clahe.apply(lab_planes[0])
            # Merge all planes.
            lab = cv2.merge(lab_planes)
            # Convert lab to bgr color space.
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return image

    def denoise_image(self,
                      image: MatLike,
                      color_mode: int,
                      user_settings: dict[str, str | int | float | None],
                      ) -> MatLike:
        """
        Apply common denoising filter to an image.

        :param image: Open cv image object.
        :param color_mode: Color mode set for processed images; color or
         grayscale.
        :param user_settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """

        default_settings = {"templateWindowSize": 7, "searchWindowSize": 21,}

        if color_mode == 0:
            # Remove filter strength value for color images and set default
            # filter strength value in settings.
            if "hColor" in user_settings:
                del user_settings["hColor"]
            default_settings["h"] = 30
            # Check for non-default user settings.
            checked_settings = self.check_defaults_overwrite(user_settings,
                                                             default_settings
                                                             )

            image = cv2.fastNlMeansDenoising(src=image,
                                             **checked_settings
                                             )
        elif color_mode == 1:
            # Remove filter strength value for grayscale images and set
            # default filter strength value in settings.
            if "h" in user_settings:
                del user_settings["h"]
            default_settings["hColor"] = 10
            # Check for non-default user settings.
            checked_settings = self.check_defaults_overwrite(user_settings,
                                                             default_settings
                                                             )

            image = cv2.fastNlMeansDenoisingColored(src=image,
                                                    **checked_settings
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

        image = cv2.bitwise_not(image)

        return image

    @staticmethod
    def rotate_image(image: MatLike,
                     user_settings: dict[str, int],
                     ) -> MatLike:
        """
        Rotate image by 90 degrees steps.

        :param image: Open cv image object.
        :param user_settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """

        # Dict of supported rotation angles.
        rotation_angles: dict[int, int] = {90: cv2.ROTATE_90_CLOCKWISE,
                                  180: cv2.ROTATE_180,
                                  270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                                  }

        # Check input rotation value.
        if user_settings["angle"] not in rotation_angles.keys():
            print("Invalid rotation value.")
            sys.exit()

        # Set rotation method.
        rotate_by = rotation_angles[user_settings["angle"]]
        # Apply image rotation.
        rotated_image = cv2.rotate(image, rotate_by)

        return rotated_image

    def sharpen_image(self,
                      image: MatLike,
                      user_settings: dict[str, str | int | float | None],
                      ) -> MatLike:
        """
        Perform sharpen or unsharp masking on an image using kernels.
        It is possible to add more kernels in future if needed.

        :param image: Open cv image object.
        :param user_settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image object for further
         manipulations.
        """

        default_settings = {"kernel": "sharpen", "filter_strength": 0.5}

        # Standard sharpening kernel from Wikipedia.
        sharpening_kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])

        # Check for non-default user settings.
        checked_settings = self.check_defaults_overwrite(user_settings,
                                                      default_settings)

        chosen_kernel = checked_settings["kernel"]
        # Applying chosen kernel to image.

        # Blend original and modified image, filter strength serves
        # as weight.
        if chosen_kernel == "sharpen":
            apply_kernel = cv2.filter2D(image, -1, sharpening_kernel)
            image = cv2.addWeighted(src1=image,
                                    alpha=1-checked_settings["filter_strength"],
                                    src2=apply_kernel,
                                    beta=checked_settings["filter_strength"],
                                    gamma=0)

        elif chosen_kernel == "unsharp_mask":
            # Apply Gaussian blur, 5x5 kernel.
            blurred_image = cv2.GaussianBlur(image, (5,5),1.0)
            # Subtract blurred image from original image.
            unsharp_image = cv2.addWeighted(src1=image,
                                            alpha=1 - checked_settings["filter_strength"],
                                            src2=blurred_image,
                                            beta=checked_settings["filter_strength"],
                                            gamma=0)

        return image

    def thresholding(self,
                     image: MatLike,
                     color_mode: int,
                     user_settings: dict[str, str | int | float | None],
                     ) -> MatLike:
        """
        Create black and white images using adaptive thresholding.

        :param image: Open cv image object.
        :param color_mode: Color mode set for processed images; color or
         grayscale.
        :param user_settings: User settings passed from settings.toml file.
        :return: Numpy array overwriting original image_object for further
         manipulations.
        """

        default_settings = {"adaptiveMethod": 0,
                          "maxValue": 255,
                          "blockSize": 199,
                          "C": 40,
                          "thresholdType": cv2.THRESH_BINARY
                          }
        passed_params = self.check_defaults_overwrite(user_settings,
                                                      default_settings)

        # Check color space in config, change color space to grayscale if
        # needed.
        if color_mode == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.adaptiveThreshold(
            src=image,
            **passed_params)

        return image

    def write_processed_image(self,
                              image: MatLike,
                              output_dir,
                              user_settings,
                              ) -> None:
        """
        Write Open cv object as image file of type chosen by user: jpg or png.

        :param image: Open cv object for writing as file.
        :param output_dir: Directory to write image files.
        :param user_settings: User settings passed from settings.toml file.
        :return: None, writes image file of chosen file type.
        """

        default_settings = {"file_extension": "jpg",
                          "quality": 85}
        checked_settings = self.check_defaults_overwrite(user_settings, default_settings)

        file_name = (f"Image_{str(self.file_counter).zfill(4)}."
                     f"{checked_settings["file_extension"]}")
        quality: list[int | dict[str, int]] = [int(
            cv2.IMWRITE_JPEG_QUALITY), int(checked_settings["quality"])
                                                 ]

        # Change jpeg quality to png compression if file_extension == png.
        if checked_settings["file_extension"] == "png":
            quality: list[int | dict[str, int]] = [
                int(cv2.IMWRITE_PNG_COMPRESSION),
                int(checked_settings["quality"])
            ]
            # Check if provided png compression factor is correct when not
            # using default value.
            if quality is not None:
                assert quality[1] in range(0, 10), "Compression value should " \
                                                "be integer between 0 and 9."

        # Check if provided jpg quality value is correct when not using
        # default value.
        if checked_settings["file_extension"] == "jpg" and quality is not None:
            assert quality[1] in range(0, 101), "Quality value should be " \
                                             "integer between 0 and 100."

        # Check for existing output directory.
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # Write Open cv object as image file.
        try:
            cv2.imwrite(os.path.join(output_dir, file_name),
                        image,
                        quality
                        )
            print(f"{file_name} file created.")
            self.file_counter += 1
        # In case of typo in provided file_extension.
        except cv2.error as e:
            print(f"Probably invalid file extension. Choose jpg or png. \n "
                  f"Error message:\n {e}")
