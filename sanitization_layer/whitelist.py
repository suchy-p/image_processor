"""
Dict containing whitelisted input types and values.
After adding new functionality it's functions' parameters need to be applied
here.
"""

# for v add types for args values, allowed values where needed.
# for each key's value add another value representing failed assertion message.
# use tuples.
whitelist = {'color_mode': 'color',
             'paths': {'input_dir': '~/Leon (Kopia)',
                       'output_dir': 'processed_images'},
             'processes': {'adjust_brightness_and_contrast':
                               {'alpha': 'None',
                                'beta': 'None'},
                           'bilateral_filter':
                               {'d': 'None',
                                'sigmaColor': 'None',
                                'sigmaSpace': 'None'},
                           'clahe':
                               {'clipLimit': 'None',
                                'tileGridSize': 'None'},
                           'denoise_filter':
                               {'h': 'None',
                                'hColor': 'None',
                                'searchWindowSize': 'None',
                                'templateWindowSize': 'None'},
                           'reverse_colors':
                               {},
                           'rotate_image':
                               {'angle': 90},
                           'sharpen_image':
                               {'filter_strength': 'None',
                               'kernel': 'unsharp_mask'},
                           'stress':
                               {'convert_to_grayscale': False,
                                'gamma': 1.25,
                                'iterations': 10,
                                'radius': 'None',
                                'sampling': 3},
                           'thresholding':
                               {'C': 'None',
                                'adaptiveMethod': 'None',
                                'blockSize': 'None',
                                'maxValue': 'None',
                                'thresholdType': 'None'},
                           },
             'write_output': {'write_image':
                                  {'file_extension': 'jpg',
                                  'quality': 80},
                              'write_pdf':
                                  {'pdf_compression': 'None',
                                   'pdf_name': 'Document'},
                              }
             }
