import cv2
from cv2.typing import MatLike
import numpy as np

from stress.functions import (
    calculate_envelopes,
    calculate_stress,
    draw_random_samples,
    get_sample_values,
)

from image_processor import ImageProcessor


check_defaults_overwrite  = ImageProcessor.check_defaults_overwrite


def stress_pipeline(image: MatLike,
                    user_settings: dict[str, str | int | float | None],
                    ) -> MatLike:

    new_image = []
    # Get image height and width values.
    # Needed for calculating default radius.
    height, width = image.shape[:2]
    # Set default params.
    default_settings = {"iterations": 10,
                      "radius": np.sqrt(height ** 2 + width ** 2).astype(int),
                      "sampling": 30,
                      "convert_to_grayscale": False
                      }

    # Check if user input overwrites default params.
    checked_settings = check_defaults_overwrite(user_settings, default_settings)
    # Set iterations from passed_params.
    iterations = checked_settings["iterations"]

    # Check grayscale conversion.
    if checked_settings["convert_to_grayscale"]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Set channels num. Dependent on color mode check above.
    channels = 3 if len(image.shape) == 3 else 1

    # Go through pipeline for set number of iterations.
    for iteration in range(iterations):
        print("Stress iteration: ", iteration + 1)

        # Get random samples for each pixel in an image.
        random_samples_coords = draw_random_samples(height=height,
                                                    width=width,
                                                    user_settings=checked_settings
                                                    )

        # Get random samples value for all color channels of each pixel.
        sample_values = get_sample_values(image=image,
                                          channels=channels,
                                          random_samples_coords=random_samples_coords
                                          )

        # Get envelope values for all coordinates.
        envelopes = calculate_envelopes(channels=channels,
                                        sample_values=sample_values
                                        )

        # Calculate STRESS.
        stress = calculate_stress(image=image,
                                  envelopes=envelopes,
                                  user_settings=checked_settings
                                  )

        # Append stress computations from an iteration.
        new_image.append(stress)

    # Create an array from new_image.
    # Reorganize new_image like arrays from functions module.
    # From: iterations, height, width, channels
    # To: height, width, channels, iterations
    new_image = np.moveaxis(np.array(new_image), 0, -1)

    # Calculate mean values from all iterations, normalize to 8-bit image.
    print("Calculating means")
    new_image = (np.mean(new_image, axis=-1)*255).astype(np.uint8)

    print("Done")
    return new_image
