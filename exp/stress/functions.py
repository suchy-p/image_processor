import numpy as np


def get_image_shape(image: np.ndarray):
    image_shape = np.shape(image)
    height, width = image_shape
    list_for_new_array = []
    for h in range(height):
        for w in range(width):
            list_for_new_array.append((h, w))

    return tuple(list_for_new_array)


def get_random_samples(height, width, radius, samples):
    height, width = height, width
    radius = radius
    samples = samples
    random_samples = []

    for h in range(height):
        for w in range(width):
            height_sample = np.random.randint(
                low= 0 if h == 0 or h - radius <= 0 else h -
                                                         radius,
                high=h + radius if radius >= h else h,
                size=samples,
                dtype=int
            )
            width_sample = np.random.randint(
                low= 0 if w == 0 or w - radius <= 0 else w -
                                                         radius,
                high=w + radius if radius >= w else w,
                size=samples,
                dtype=int
            )
            samples.append(tuple(zip(height_sample, width_sample)))

    return random_samples


def get_envelopes(image, random_samples):
    image = image
    random_samples = random_samples

    red_channel = set()
    green_channel = set()
    blue_channel = set()

    greyscale = set()

    min_envelope = list()
    max_envelope = list()

    if len(image[(0,0)]) == 3:
        for random_sample in random_samples:
            r, g, b = image[random_sample]
            red_channel.add(r)
            green_channel.add(g)
            blue_channel.add(b)

        min_envelope = (min(red_channel), min(green_channel), min(blue_channel)
                        )
        max_envelope = (max(red_channel), max(green_channel), max(blue_channel)
                        )

    elif len(image[(0,0)]) == 1:
        for random_sample in random_samples:
            grey = int(image[random_sample])
            greyscale.add(grey)

        min_envelope = (min(greyscale),)
        max_envelope = (max(greyscale),)

    return min_envelope, max_envelope


def apply_stress(pixel, samples):
    pixel = pixel
    e_min, e_max = get_envelopes(pixel, samples)
    stress = []

    for channel in pixel:
        # p = p_0 - E_min / E_max - E_min
        numerator = channel - e_min
        denominator = e_max - e_min

        try:
            channel_stress = np.abs(np.divide(numerator, denominator))
            stress.append(int(channel_stress*255))
        except ValueError as e:
            if str(e) == "cannot convert float NaN to integer":
                stress.append(0)
            else:
                print(e)

    return stress
