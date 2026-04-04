import numpy as np

from .functions import (get_image_shape, get_envelopes,
                        get_random_samples, apply_stress)


def stress_pipeline(image: np.ndarray,
                    radius: int,
                    samples: int,
                    ):
    stress_image = []
    height, width = get_image_shape(image)
    random_samples = get_random_samples(height, width,
                                        radius,
                                        samples)

    for h in range(height+1):
        for w in range(width+1):
            pixel = image[(h, w)]
            stress_pixel = apply_stress(pixel, random_samples)
            stress_image.append(stress_pixel)

    return np.array(stress_image)