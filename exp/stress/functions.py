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


def get_sample_values(image: MatLike,
                      channels: int,
                      random_samples_coords: np.ndarray
                      ) -> np.ndarray:
    """
    Retrieves pixel intensity values for a matrix of sampling coordinates.

    Uses vectorized indexing to extract pixel values from the source image for
    each pre-calculated coordinate.

    :param image: Source image as a NumPy array.
    :param channels: Number of image channels (1 or 3).
    :param random_samples_coords: NumPy array of coordinates to sample from.
    :return: A NumPy array containing the sampled values.
    """
    y_axis = random_samples_coords[..., 0]
    x_axis = random_samples_coords[..., 1]
    all_samples = image[y_axis, x_axis]
    # In rgb image transpose all_samples, so samples are on third axis,
    # for the sake of arrays shape continuity: height, width, channels, other
    if channels == 3:
        all_samples = all_samples.transpose(0, 1, 3, 2)

    return all_samples


def calculate_envelopes(channels: int,
                        sample_values: np.ndarray
                        ) -> np.ndarray:
    """
    Calculates local minimum and maximum values (envelopes) for each pixel.

    Processes collected samples to determine the dynamic range (min/max) in the
    pixel's local neighborhood. Values are normalized to the 0.0-1.0 range.

    :param channels: Number of color channels.
    :param sample_values: Sample values obtained from get_sample_values.
    :return: A NumPy array of shape (height, width, [channels], 2) containing
             [min, max] values for each pixel and channel.
    """

    local_min = None
    local_max = None

    if channels == 1:
        local_min = np.min(sample_values, axis=2).astype(float)/255.0
        local_max = np.max(sample_values, axis=2).astype(float)/255.0

    elif channels == 3:
        local_min = np.min(sample_values, axis=3).astype(float)/255.0
        local_max = np.max(sample_values, axis=3).astype(float)/255.0

    envelopes = np.stack([local_min, local_max], axis=-1)

    return envelopes


def calculate_stress (image: MatLike,
                      envelopes: np.ndarray,
                      gamma: float = 1.0,
                      ) ->  np.ndarray:
    """
    Performs the STRESS transformation on the image based on local envelopes.

    Normalizes pixel values relative to local minimum and maximum envelopes.
    Includes gamma correction and internal normalization to a 0.0-1.0 range.

    :param image: Original input image.
    :param envelopes: Matrix of local envelopes [min, max] for each pixel.
    :param gamma: Gamma correction factor (default 1.0).
    :return: Processed image as a floating-point NumPy array (0.0 to 1.0).
    """
    # Normalize image for 0.0 - 1.0 range.
    image = image.astype(float)/255.0
    min_envelopes = envelopes[..., 0]
    max_envelopes = envelopes[..., 1]

    numerator = image - min_envelopes
    denominator = max_envelopes - min_envelopes

    # Calculating stress; out=image.astype(float) as fallback for zero
    # division handling.
    stress =  np.divide(numerator, denominator,
                       out=image,
                       dtype=float,
                       where=denominator!=0
                        )
    # Normalize values.
    stress = np.clip(stress, 0, 1)
    # Gamma correction, if needed.
    new_image = np.power(stress, 1.0/gamma)

    return new_image
