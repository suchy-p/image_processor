from os.path import expanduser

import cv2
from cv2.typing import MatLike
import numpy as np

from exp.stress.functions import (
    calculate_envelopes,
    calculate_stress,
    draw_random_samples,
    get_file_list,
    get_sample_values,
)


input_dir = expanduser("~/Leon (Kopia)")
radius = 1169
sampling = 10
iterations = 20

files_to_process = get_file_list(input_dir)


for file in files_to_process:
    # Open image as Open cv object
    image: MatLike = cv2.imread(file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_image = []
    # Get image dimensions.
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    for iteration in range(iterations):
        print("Iteration: ", iteration + 1)

        # Get random samples for each pixel in an image.
        random_samples_coords = draw_random_samples(height=height,
                                                    width=width,
                                                    radius=radius,
                                                    sampling=sampling
                                                    )

        # Get random samples value for each color channel.
        sample_values = get_sample_values(image=image,
                                          channels=channels,
                                          random_samples_coords=
                                          random_samples_coords
                                          )

        # Get envelope values for each coordinate.
        envelopes = calculate_envelopes(channels=channels,
                                        sample_values=sample_values
                                        )

        # Calculate stress.
        stress = calculate_stress(image=image,
                                  envelopes=envelopes
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
    new_image = np.array(new_image).reshape((height,
                                             width,
                                             channels,
                                             iterations)
                                            )

    # Calculate mean values from all iterations.
    print("Calculating means")
    new_image = np.mean(new_image, axis=-1)*255
    # Reshape array to shape of output image, write file.
    new_image = new_image.reshape((height, width, channels)).astype(np.uint8)
    new_image = cv2.imwrite(file, new_image)
    print("Done")

