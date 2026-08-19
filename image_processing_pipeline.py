import os

import cv2
from cv2.typing import MatLike

from image_processor import ImageProcessor
from stress.stress_pipeline import stress_pipeline


class ImageProcessingPipeline:
    """
    Runs processes enabled in configuration dict, passes params to
     selected processes.
    :return: None, applies selected processes to image files.
    """
    def __init__(self, settings: dict):
        self.processes = ImageProcessor(settings)
        self.stress_pipeline = stress_pipeline
        self.settings = settings
        # Since ImageProcessor also needs input and output dir and I need to
        # instantiate it anyway, self.input_dir and self.output dir are
        # derived from there.
        self.input_dir = self.processes.input_dir
        self.output_dir = self.processes.output_dir
        self.color_mode_map = {
            "color": cv2.IMREAD_COLOR,
            "grayscale": cv2.IMREAD_GRAYSCALE,
        }
        self.color_mode_set = self.color_mode_map[settings["colors"]["color_mode"]]
        # Remove file counter from ImageProcessor after moving file writing
        # to separate module. Also reminded in ImageProcessor.
        self.file_counter = 1

    def run_selected_processes(self) -> None:
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
            image_object: MatLike = cv2.imread(image, self.color_mode_set)

            # Check if given functionality is enabled_processes in
            # settings.toml.

            if enabled_processes["rotate_image"]["enabled"]:
                image_object = self.processes.rotate_image(
                    image_object, settings=enabled_processes["rotate_image"])

            if enabled_processes["reverse_colors"]["enabled"]:
                image_object = self.processes.reverse_color(image_object)

            if enabled_processes["clahe"]["enabled"]:
                image_object = self.processes.clahe(
                    image_object,
                    color_mode=self.color_mode_set,
                    settings=enabled_processes["clahe"]
                )

            if enabled_processes["adjust_brightness_and_contrast"]["enabled"]:
                image_object = self.processes.adjust_brightness_and_contrast(
                    image_object, settings=enabled_processes[
                        "adjust_brightness_and_contrast"])

            if enabled_processes["sharpen_image"]["enabled"]:
                image_object = self.processes.sharpen_image(
                    image_object, settings=enabled_processes["sharpen_image"])

            if enabled_processes["bilateral_filter"]["enabled"]:
                image_object = self.processes.bilateral_filter(
                    image_object, settings=enabled_processes[
                        "bilateral_filter"])

            if enabled_processes["denoise_filter"]["enabled"]:
                image_object = self.processes.denoise_image(
                    image_object,
                    color_mode=self.color_mode_set,
                    settings=enabled_processes["denoise_filter"]
                )

            if enabled_processes["stress"]["enabled"]:
                image_object = self.stress_pipeline(image_object,
                                                    settings=enabled_processes[
                                                        "stress"])

            if enabled_processes["thresholding"]["enabled"]:
                image_object = self.processes.thresholding(
                    image_object, color_mode=self.color_mode_set,
                    settings=enabled_processes[
                        "thresholding"])

            if enabled_processes["write_image"]["enabled"]:
                self.processes.write_processed_image(
                    image_object,
                    output_dir=self.output_dir,
                    settings=enabled_processes["write_image"]
                )

        if enabled_processes["write_pdf"]["enabled"]:
            params = self.processes.check_defaults_overwrite(enabled_processes["write_pdf"])
            self.processes.write_pdf_file(**params)

        # Reset file counter after all files in dir have been processed.
        self.file_counter = 1
