from image_processor import ImageProcessor

input_dir = ""
output_dir = ""

config = {'color_space': 'color',
          'write_processed_image': [True,
                                {'file_extension': 'jpg',
                                 'quality': 9
                               }
                                    ],
          'write_pdf_file': [False,
                             {'images_path': output_dir,
                              'images_file_type': 'png',
                              'pdf_file_name': 'Tygodnik '
                                              'Rolniczo-Przemysłowy',
                              'pdf_file_compression': None
                              }
                             ],
          'bilateral_filter': [False,
                               {'d': None,
                                'sigma_color': None,
                                'sigma_space': None
                                }
                               ],
          'black_and_white': [False,
                              {'method': 'gaussian',
                               'max_value': None,
                               'block_size': None,
                               'constant': None
                               }
                              ],
          'clahe': [True,
                    {'clip_limit': None,
                     'tile_grid_size': None
                     }
                    ],
          'contrast_brightness': [False,
                                  {'alpha': None,
                                  'beta': None
                                   }
                                  ],
          'stress': [True, {'iterations': 4,
                            'radius': 1500,
                            'samples': 3}],
          'denoise_image': [False, {'filter_strength': 10}],
          'reverse_colors': [False],
          'run_rotate_adjustment': [False, {'rotation_angle': 90}],
          'sharpen_image': [False, {'kernel': 'unsharp_mask',
                                   'strength': None
                                    }
                            ],
          }

if __name__ == '__main__':
    image_processor = ImageProcessor(input_dir=input_dir,
                                     output_dir=output_dir)
    image_processor.image_processing_pipeline(**config)
