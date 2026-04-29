from os import chdir
from os.path import expanduser

import cv2
from cv2.typing import MatLike
import numpy as np

from exp.stress.functions import (get_file_list, draw_random_samples,
                                  get_sample_values, calculate_envelopes,
                                  calculate_stress)


input_dir = expanduser("~/Leon (Kopia)")
radius = 512
sampling = 5
iterations = 100

files_to_process = get_file_list(input_dir)
chdir(input_dir)
# Apply selected processes to each image
for file in files_to_process:
    # Open image as Open cv object
    image: MatLike = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_image = []
    # Get image dimensions.
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    # Get coordinates of each pixel in an image.
    coordinates = tuple((y, x) for y in range(height) for x in range(width))

    for iteration in range(iterations):
        print("Iteration: ", iteration + 1)

        random_samples_coords = []
        # Get random samples coords.
        for coord in coordinates:
            random_samples_coords.append(draw_random_samples(height=height,
                                                             width=width,
                                                             radius=radius,
                                                             sampling=sampling,
                                                             coord=coord
                                                             )
                                  )
        random_samples_coords = tuple(random_samples_coords)

        # Get random samples value for each color channel.
        sample_values = get_sample_values(image=image,
                                          height=height,
                                          width=width,
                                          channels=channels,
                                          sampling=sampling,
                                          random_samples_coords=
                                          random_samples_coords
                                          )

        # Get envelope values for each coordinate.
        envelopes = calculate_envelopes(height=height,
                                        width=width,
                                        channels=channels,
                                        sample_values=sample_values
                                        )

        # Calculate stress.
        stress = calculate_stress(image=image,
                                  height=height,
                                  width=width,
                                  channels=channels,
                                  envelopes=envelopes
                                  )

        # Append image stress computations for each iteration.
        if new_image is None:
             new_image = list(stress)
        else:
            # new_image = new_image + [list(stress)]
            new_image.append(stress)

    # Transpose new_image to organize arguments as in .functions, for sake of
    # consistency: iterations, height, width, channels
    # -> height, width, channels, iterations
    new_image  =np.transpose(new_image, (1, 2, 3, 0))
    new_image = np.array(new_image).reshape((height, width, channels,
                                             iterations)
                                            )

    # Calculate mean for each color channel for every pixel in an image.
    print("Calculating means")
    new_image = np.mean(new_image, axis=3)
    # Reshape array to shape of output image, write file.
    new_image = new_image.reshape((height, width, channels)).astype(np.uint8)
    new_image = cv2.imwrite(file, new_image)
    print("Done")

