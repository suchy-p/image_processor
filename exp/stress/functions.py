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

def draw_random_samples(height: int,
                        width: int,
                        radius: int,
                        sampling: int,
                        ) -> np.ndarray:
    """
    Generates a matrix of random sampling coordinates for every pixel in the
    image.

    For each pixel, a set of random coordinates is generated within a radius.
    The samples are automatically constrained within the image boundaries.

    :param height: Image height.
    :param width: Image width.
    :param radius: Radius within which to sample.
    :param sampling: Number of samples to draw per pixel.
    :return: A NumPy array of shape (height, width, sampling, 2) containing
    (y, x) pairs.
    """

    # Generate image coordinate axes.
    # Shapes: y_axis = height, 1; x_axis = 1, width
    y_axis, x_axis = np.ogrid[0:height, 0:width]

    # Set low and high boundaries for sampling radius of each pixel.
    min_height = np.where(y_axis - radius > 0,
                          y_axis - radius, 0
                          )
    max_height = np.where(y_axis + radius <= height,
                          y_axis + radius, height
                          )
    min_width = np.where(x_axis - radius > 0,
                         x_axis - radius, 0
                         )
    max_width = np.where(x_axis + radius <= width,
                         x_axis + radius, width
                         )

    # Reshape height and width for stacking, adding extra dimension ensures
    # that h, w and sampling fit into right places in this order:
    # height =  h, 1, s; width = 1, w, s;
    # broadcasting from right to left: s1h and sw1
    min_height = min_height[:, :, np.newaxis]
    max_height = max_height[:, :, np.newaxis]
    min_width = min_width[:, :, np.newaxis]
    max_width = max_width[:, :, np.newaxis]

    # Get random sampling coords.
    height_coords = np.random.randint(low=min_height, high=max_height,
                                      size=(height, width, sampling))
    width_coords = np.random.randint(low=min_width, high=max_width,
                    size=(height, width, sampling))

    # Stack coords
    random_samples_coords = np.stack([height_coords, width_coords],
                                     axis=-1)

    return random_samples_coords
