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
                    settings: dict[str, str | int | float | None],
                    ) -> MatLike:

    new_image = []
    # Get image dimensions.
    height, width = image.shape[:2]
    channels = 3 if len(image.shape) == 3 else 1
    default_params = {"iterations": 10,
                      "radius": height if height >= width else width,
                      "convert_to_grayscale": False
                      }
    passed_params = check_defaults_overwrite(settings, default_params)
    iterations = int(passed_params["iterations"])

    convert_to_grayscale = passed_params["convert_to_grayscale"]

    for iteration in range(iterations):
        radius = int(passed_params["radius"])
        print("Stress iteration: ", iteration + 1)
        # Get random samples for each pixel in an image.
        random_samples_coords = draw_random_samples(height=height,
                                                    width=width,
                                                    # radius=radius,
                                                    settings=settings
                                                    )

        # Get random samples value for each color channel.
        sample_values = get_sample_values(
            image=image,
            channels=channels,
            random_samples_coords=random_samples_coords
        )

        # Get envelope values for each coordinate.
        envelopes = calculate_envelopes(channels=channels,
                                        sample_values=sample_values
                                        )

        # Calculate stress.
        stress = calculate_stress(image=image,
                                  envelopes=envelopes,
                                  settings=settings
                                  )

        # Append image stress computations for each iteration.
        if new_image is None:
            new_image = list(stress)
        else:
            new_image.append(stress)

    # Transpose new_image to organize arguments as in .functions, for sake of
    # consistency: iterations, height, width, channels
    # -> height, width, channels, iterations
    new_image = np.moveaxis(np.array(new_image), 0, -1)
    new_image = np.array(new_image).reshape((height, width, channels,
                                             iterations)
                                            )

    # Calculate mean values from all iterations.
    print("Calculating means")
    new_image = np.mean(new_image, axis=-1)*255
    # Reshape array to shape of output image, write file.
    new_image = new_image.reshape((height, width, channels)).astype(np.uint8)

    if convert_to_grayscale:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    print("Done")
    return new_image
