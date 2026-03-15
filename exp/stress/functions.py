import numpy as np
import cv2


def get_image_shape(image: np.ndarray):
    image_shape = np.shape(image)
    height, width = image_shape
    list_for_new_array = []
    for h in range(height):
        for w in range(width):
            list_for_new_array.append((h, w))

    return tuple(list_for_new_array)


def get_random_samples(image, sampling_range, sample_size):
    height, width = np.shape(image)
    sampling_range = sampling_range
    sample_size = sample_size
    samples = []

    for h in range(height):
        for w in range(width):
            height_sample = np.random.randint(
                low= 0 if h == 0 or h - sampling_range <= 0 else h -
                                                                sampling_range,
                high= h + sampling_range if sampling_range >= h else h,
                size=sample_size,
                dtype=int
            )
            width_sample = np.random.randint(
                low= 0 if w == 0 or w - sampling_range <= 0 else w -
                                                                sampling_range,
                high= w + sampling_range if sampling_range >= w else w,
                size=sample_size,
                dtype=int
            )
            samples.append(tuple(zip(height_sample, width_sample)))

    return samples


def get_envelopes(image, samples):
    image = image
    samples = samples

    red_channel = set()
    green_channel = set()
    blue_channel = set()

    greyscale = set()

    min_envelope = list()
    max_envelope = list()

    if len(image[(0,0)]) == 3:
        for sample in samples:
            r, g, b = image[sample]
            red_channel.add(r)
            green_channel.add(g)
            blue_channel.add(b)

        min_envelope = (min(red_channel), min(green_channel), min(blue_channel)
                        )
        max_envelope = (max(red_channel), max(green_channel), max(blue_channel)
                        )

    elif len(image[(0,0)]) == 1:
        for sample in samples:
            grey = int(image[sample])
            greyscale.add(grey)

        min_envelope = (min(greyscale),)
        max_envelope = (max(greyscale),)

    return min_envelope, max_envelope


def apply_stress(image, samples, h, w):
    h = h
    w = w
    pixel = image[(h, w)]
    e_min, e_max = get_envelopes(pixel, samples)
    samples = samples
    stress = []

    for channel in pixel:
        # g = (channel-b) * (w-b) / |w-b|**2 <== to konwersja rgb 2 gr
        # tu ma być local color correction:
        # p = p_0 - E_min / E_max - E_min

        numerator = channel - e_min
        denominator = e_max - e_min
        # b, w = get_envelopes(image, samples)
        # channel_minus_b = np.subtract(p, b)
        # w_minus_b = np.subtract(w, b)
        # numerator = np.dot(channel_minus_b, w_minus_b)
        # denominator = np.absolute(np.dot(w_minus_b, w_minus_b))

        try:
            channel_stress = numerator / denominator
            channel_stress = np.abs(np.divide(numerator, denominator))
            stress.append(int(channel_stress*255))
        except ValueError as e:
            if str(e) == "cannot convert float NaN to integer":
                stress.append(0)
            else:
                print(e)

    return stress
