import numpy as np
import pytest
from random import randint
from exp.stress.functions import (apply_stress, get_image_shape,
                                  draw_random_samples, get_envelopes)


def test_get_image_shape():
    test_arrays = [np.zeros((2, 3)),
                   np.zeros((10, 25)),
                   np.zeros((1400, 500)),
                   np.zeros((3000, 1000))
                   ]

    for array in test_arrays:
        array_vals = []
        for a in range(array.shape[0]):
            for b in range(array.shape[1]):
                array_vals.append((a, b))
        array_vals = tuple(array_vals)
        assert get_image_shape(array) == array_vals


def test_get_random_samples():
    test_image = np.zeros((140, 50))
    test_sampling_ranges = (20, 80, 140)
    test_samples = (3, 10, 20)

    for sampling_range in test_sampling_ranges:
        for sample in test_samples:
            res = draw_random_samples(test_image, sampling_range, sample)
            assert len(res[0]) == sample

# @pytest.mark.skip(reason="Not implemented")
def test_apply_stress():
    image_shape = (140, 50)
    test_image = np.random.randint(0,255, size=(*image_shape, 3))
    test_image_black = np.zeros_like(test_image)
    test_image_white = np.full_like(test_image, 255)

    h_samples = np.random.randint(0, 141, size=3)
    w_samples = np.random.randint(0, 51, size=3)
    samples = tuple(zip(h_samples, w_samples))

    for h in range(image_shape[0]):
        for w in range(image_shape[1]):
            res = apply_stress(test_image, samples, h, w)
            for i in res:
                assert 0 <= int(i) <= 255
            # assert (0 <= int(i) <= 255 for i in res)
            res_black = apply_stress(test_image_black, samples, h, w)
            for i in res_black:
                assert int(i) == 0
            res_white = apply_stress(test_image_white, samples, h, w)
            for i in res_white:
                assert int(i) == 0


def test_get_envelopes():
    image_shape = (140, 50)
    test_image_rgb = np.random.randint(0,255, size=(*image_shape, 3)
                                       )
    test_image_grey = np.random.randint(0, 255, size=(*image_shape, 1)
                                        )
    h_samples = np.random.randint(0, 140, size=3)
    w_samples = np.random.randint(0, 50, size=3)
    samples = tuple(zip(h_samples, w_samples))

    for h in range(image_shape[0]):
        for w in range(image_shape[1]):
            pixel_rgb = test_image_rgb[(h, w)]
            res_rgb = get_envelopes(pixel_rgb,test_image_rgb, samples)
            assert len(res_rgb[0]) == 3 and len(res_rgb[1]) == 3
            assert min(res_rgb[0]) >=0 and max(res_rgb[0]) <= 255
            assert min(res_rgb[1]) >=0 and max(res_rgb[1]) <= 255

            pixel_grey = test_image_grey[(h, w)]
            res_grey = get_envelopes(pixel_grey,test_image_grey, samples)
            assert len(res_grey[0]) == 1 and len(res_grey[1]) == 1
            assert min(res_grey[0]) >=0 and max(res_grey[0]) <= 255
            assert min(res_grey[1]) >=0 and max(res_grey[1]) <= 255
