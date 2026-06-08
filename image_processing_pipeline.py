import os

import cv2
from cv2.typing import MatLike

from image_processor import ImageProcessor
from exp.stress.pipeline import stress_pipeline


class ImageProcessingPipeline:
    """
    Runs processes enabled in configuration dict, passes params to
     selected processes.
    :return: None, applies selected processes to image files.
    """
    def __init__(self, settings):
        self.processes = ImageProcessor(settings)
        self.stress_pipeline = stress_pipeline
        self.settings = settings
        self.input_dir = os.path.expanduser(settings["paths"]["input_dir"])
        self.output_dir = os.path.join(self.input_dir, settings["paths"][
            "output_dir"])
        self.color_space_map = {
            "color": cv2.IMREAD_COLOR,
            "grayscale": cv2.IMREAD_GRAYSCALE,
        }
        self.color_space_set = self.color_space_map[settings["color_space"]]
        self.file_counter = 1

    def run_selected_processes(self):
        # Create list of images for processing.
        os.chdir(self.input_dir)
        to_process: list[str] = [
            os.path.abspath(item) for item in os.listdir(os.getcwd())
            if os.path.isfile(item)
        ]
        print(to_process)
        enabled_processes = (self.settings["processes"] |
                             self.settings["write_output"])

        # Apply selected processes to each image
        for image in to_process:
            print(f"Processing {self.file_counter} image of "
                  f"{len(to_process)}: \n"
                  f"\t{os.path.basename(image)}")

            # Open image as Open cv object
            image_object: MatLike = cv2.imread(image, self.color_space_set)

            # Check if given functionality is enabled_processes in
            # settings.toml.

            if enabled_processes["rotate_image"]["enabled"]:
                params = self.processes.checker(
                    enabled_processes["rotate_image"])
                image_object = self.processes.rotate_image(
                    image_object, **params)

            if enabled_processes["reverse_colors"]["enabled"]:
                image_object = self.processes.reverse_color(image_object)

            if enabled_processes["clahe"]["enabled"]:
                params = self.processes.checker(enabled_processes["clahe"])
                image_object = self.processes.clahe(
                    image_object,
                    color_space=self.color_space_set,
                    **params
                )

            if enabled_processes["adjust_brightness_and_contrast"]["enabled"]:
                params = self.processes.checker(
                    enabled_processes["adjust_brightness_and_contrast"])
                image_object = self.processes.contrast_brightness(
                    image_object, **params)

            if enabled_processes["sharpen_image"]["enabled"]:
                params = self.processes.checker(
                    enabled_processes["sharpen_image"])
                image_object = self.processes.sharpen_image(
                    image_object, **params)

            if enabled_processes["bilateral_filter"]["enabled"]:
                params = self.processes.checker(
                    enabled_processes["bilateral_filter"])
                image_object = self.processes.bilateral_filter(
                    image_object, **params)

            if enabled_processes["denoise_filter"]["enabled"]:
                params = self.processes.checker(
                    enabled_processes["denoise_filter"])
                image_object = self.processes.denoise_image(
                    image_object,
                    color_space=self.color_space_set,
                    **params
                )

            if enabled_processes["stress"]["enabled"]:
                params = self.processes.checker(enabled_processes["stress"])
                image_object = self.stress_pipeline(image_object, **params)

            if enabled_processes["thresholding"]["enabled"]:
                params = self.processes.checker(
                    enabled_processes["thresholding"])
                image_object = self.processes.thresholding(
                    image_object, **params)

            if enabled_processes["write_image"]["enabled"]:
                params = self.processes.checker(
                    enabled_processes["write_image"])
                self.processes.write_processed_image(
                    image_object,
                    output_dir=self.output_dir,
                    **params
                )

        if enabled_processes["write_pdf"]["enabled"]:
            params = self.processes.checker(enabled_processes["write_pdf"])
            self.processes.write_pdf_file(**params)

        # Reset file counter after all files in dir have been processed.
        self.file_counter = 1
