from cv2.typing import MatLike
import numpy as np


"""
Functions used in pipeline implementing Spatio-Temporal Retinex-Inspired Envelope with Stochastic Sampling (STRESS).
Paper explaining idea behind STRESS and its use cases:
O. Kolås, I. Farup, A. Rizzi, Spatio-Temporal Retinex-Inspired Envelope with Stochastic Sampling: A Framework for 
Spatial Color Algorithms, "Journal of Imaging Science and Technology" 55(4): 040503-1–040503-10, 2011.
Shortened link: https://tiny.pl/z3bvw-w6k
"""

def draw_random_samples(height: int,
                        width: int,
                        user_settings: dict[str, str | int | float | None],
                        ) -> np.ndarray:
    """
    Generates a matrix of random sampling coordinates for every pixel in the image.

    For each pixel, a set of random coordinates is generated within a radius.
    The samples are automatically constrained within the image boundaries.

    :param height: Image height.
    :param width: Image width.
    :param user_settings: User settings.
    :return: A NumPy array of shape (height, width, sampling, coords), where coords=2 in np.shape and contains (y, x)
    pairs pointing desired sample location.
    """

    radius = user_settings["radius"]
    sampling = user_settings["sampling"]
    # Generate image coordinate axes.
    # Shapes: y_axis = height, 1; x_axis = 1, width
    y_axis, x_axis = np.ogrid[0:height, 0:width]

    # Set low and high boundaries for sampling radius of each pixel.
    # Reminder: max_height and max_width are later passed as arguments when creating arrays, where upper values are
    # non-inclusive.
    min_height = np.where(y_axis - radius > 0,
                          y_axis - radius,
                          0
                          )
    max_height = np.where(y_axis + radius <= height,
                          y_axis + radius,
                          height
                          )
    min_width = np.where(x_axis - radius > 0,
                         x_axis - radius, 0
                         )
    max_width = np.where(x_axis + radius <= width,
                         x_axis + radius,
                         width
                         )

    # Reshape height and width for stacking with sampling.
    # Adding extra dimension ensures that h, w and sampling fit into right places in this order:
    # height =  h, 1 -> h, 1, s; width = 1, w -> 1, w, s.
    # Reminder: numpy broadcasts from right to left: s1h and sw1
    min_height = min_height[..., np.newaxis]
    max_height = max_height[..., np.newaxis]
    min_width = min_width[..., np.newaxis]
    max_width = max_width[..., np.newaxis]

    # Get random samples coords.
    height_coords = np.random.randint(low=min_height, high=max_height,
                                      size=(height, width, sampling)
                                      )
    width_coords = np.random.randint(low=min_width, high=max_width,
                                     size=(height, width, sampling)
                                     )

    # Stack coords.
    # Shape of this array: height, width, sampling, sample coords.
    random_samples_coords = np.stack([height_coords, width_coords],
                                     axis=-1)

    return random_samples_coords


def get_sample_values(image: MatLike,
                      channels: int,
                      random_samples_coords: np.ndarray
                      ) -> np.ndarray:
    """
    Uses an array of random samples' coordinates for every pixel in the image to get specified number of these samples'
    intensity values.

    :param image: Source image as a NumPy array.
    :param channels: Number of image channels (1 or 3).
    :param random_samples_coords: NumPy array of coordinates to sample from.
    :return: A NumPy array containing the sampled values.
    """

    # Extract row (y) and column (x) coordinates.
    y_coords = random_samples_coords[..., 0]
    x_coords = random_samples_coords[..., 1]
    # Get pixel intensity values from specified image coordinates.
    all_samples = image[y_coords, x_coords]
    # In RGB image transpose all_samples, so samples are on last axis,
    # for the sake of arrays shape continuity: height, width, channels, other (in this case samples' values).
    if channels == 3:
        all_samples = all_samples.transpose(0, 1, 3, 2)

    return all_samples


def calculate_envelopes(channels: int,
                        sample_values: np.ndarray
                        ) -> np.ndarray:
    """
    Calculates local minimum and maximum values (envelopes) for each pixel.

    Values are normalized to the 0.0-1.0 range.

    :param channels: Number of color channels.
    :param sample_values: Sample values obtained from get_sample_values.
    :return: A NumPy array of shape (height, width, [channels], 2) containing
             [min, max] values for each pixel and channel.
    """

    local_min = None
    local_max = None

    # Calculate envelopes for grayscale or RGB, normalize results to 0.0 - 1.0 range.
    if channels == 1:
        local_min = np.min(sample_values, axis=2).astype(float)/255
        local_max = np.max(sample_values, axis=2).astype(float)/255

    elif channels == 3:
        local_min = np.min(sample_values, axis=3).astype(float)/255
        local_max = np.max(sample_values, axis=3).astype(float)/255

    # Create an array containing envelopes for each pixel's channels.
    envelopes = np.stack([local_min, local_max], axis=-1)

    return envelopes


def calculate_stress(image: MatLike,
                     envelopes: np.ndarray,
                     user_settings: dict[str, str | int | float | None],
                     ) -> np.ndarray:
    """
    Performs the STRESS transformation on the image based on local envelopes. Normalizes pixel values to 0.0-1.0 range.
    Includes gamma correction.

    :param image: Original input image.
    :param envelopes: Array of local envelopes [min, max] for each pixel.
    :param user_settings: User settings passed from settings.toml file.
    :return: Processed image as a floating-point NumPy array (0.0 to 1.0).
    """

    gamma = user_settings["gamma"]
    # Normalize image to 0.0 - 1.0 range.
    image = image.astype(float)/255.0
    # Divide envelopes into two arrays containing min and max values.
    min_envelopes = envelopes[..., 0]
    max_envelopes = envelopes[..., 1]

    # Calculate numerator and denominator for STRESS formula.
    numerator = image - min_envelopes
    denominator = max_envelopes - min_envelopes

    # Calculating STRESS; out=image normalized as a fallback for zero
    # division handling.
    stress = np.divide(numerator, denominator,
                       out=image,
                       dtype=float,
                       where=denominator != 0
                       )
    # Clip values outside range expected for 8-bit image.
    stress = np.clip(stress, 0, 1)
    # Apply gamma correction, if set.
    new_image = np.power(stress, 1.0/gamma)

    return new_image
