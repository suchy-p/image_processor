import os

from cv2.typing import MatLike
import numpy as np


def get_file_list(input_dir: str) -> list[str]:
    """
    Retrieves a list of full paths to image files (.jpg, .png) in a directory.

    Scans the specified directory for files with JPEG and PNG extensions and
    joins them with the input directory path.

    :param input_dir: Path to the directory containing images to be processed.
    :return: List of full strings representing paths to the discovered images.
    """
    file_list = [os.path.join(input_dir, file) for file in
                 os.listdir(input_dir)
                 if os.path.isfile(os.path.join(input_dir, file))
                 and file.endswith((".jpg", ".png"))
                 ]

    return file_list
