"""
Dict containing whitelisted input types and values.
After adding new functionality it's functions' parameters need to be applied
here. This dict follows the structure of settings.toml.

Each parameter should have its own nested dict containing parameter name as
key and tuple of three items as value. First one declares expected input data
type. Second one contains lambda or lambdas (in tuple) verifying user input
values. Last one is an error message (or messages in tuple, for multiple
lambdas) displayed when given check fails.

The "check_against" key at the bottom is reserved for finite lists of values to
check against in lambdas, like characters unpermitted in file or folder
names. In verifying lambda it should be a second parameter.
"""

from os.path import exists

whitelist = {"colors": {"color_mode" :("str",
                                       lambda color_mode: color_mode in (
                                           "color", "grayscale",
                                           "greyscale"),
                                       "Proper color modes are color and "
                                       "grayscale (greyscale).")
                            },
             "paths": {"input_dir": ("str",
                                     lambda path: exists(path),
                                     "Invalid or nonexistent input directory "
                                     "path."),
                       "output_dir": ("str",
                                      lambda output_dir, unpermitted_chars:
                                      unpermitted_chars not in output_dir,
                                      "Invalid output directory path. Check "
                                      "if you are using unpermitted "
                                      "characters in output directory name.")
                       },
             "processes": {"adjust_brightness_and_contrast":
                               {"alpha": ("int",
                                          lambda alpha: 0<= alpha,
                                          "Alpha value should be an integer "
                                          "equal to or greater than 0"),
                                "beta": ("int",
                                         "Beta value should be an integer.")
                                },
                           "bilateral_filter":
                               {"d": ("int",
                                      lambda d: 0<= d,
                                      "d value should be an integer equal to "
                                      "or greater than 0."),
                                "sigmaColor": ("int",
                                               lambda sigmaColor: 0<=
                                                                  sigmaColor,
                                               "sigmaColor value should be an "
                                               "integer equal to or greater "
                                               "than 0."),
                                "sigmaSpace": ("int",
                                               lambda sigmaSpace: 0<=
                                                                  sigmaSpace,
                                               "sigmaSpace value should be an "
                                               "integer equal to or greater "
                                               "than 0.")
                                },
                           "clahe":
                               {"clipLimit": ("int",
                                              lambda clipLimit: 0 <=
                                                                 clipLimit,
                                              "clipLimit value should be an "
                                              "integer equal to or greater "
                                              "than 0."),
                                "tileGridSize": ("tuple",
                                                 lambda tileGridSize:
                                                 [num for num in tileGridSize
                                                 if num > 0 and
                                                 len(tileGridSize) == 2],
                                                "tileGridSize should be a "
                                                "tuple of two integers "
                                                "representing side of a "
                                                "square or a rectangle.")
                                },
                           "denoise_filter":
                               {"h": ("int",
                                      lambda h: 0<h,
                                      "h value should be a positive integer."),
                                "hColor": ("int",
                                           lambda hColor: 0<hColor,
                                           "hColor value should be a "
                                           "positive integer."),
                                "searchWindowSize": ("int",
                                                     lambda searchWindowSize:
                                                      searchWindowSize % 2
                                                      == 0,
                                                     "searchWindowSize should "
                                                     "be a positive odd "
                                                     "integer."),
                                "templateWindowSize": ("int",
                                                       lambda templateWindowSize:
                                                        templateWindowSize % 2
                                                        == 0,
                                                       "templateWindowSize "
                                                       "should be a positive "
                                                       "odd integer.")
                                },
                           "reverse_colors":
                               {"reverse_colors": ("bool",
                                                   lambda reverse_colors:
                                                   reverse_colors in (True,
                                                                      False),
                                                   "Reverse_colors takes "
                                                   "True or False as input.")
                                },
                           "rotate_image":
                               {"angle": ("int",
                                          lambda angle: angle in (90,
                                                                   180,
                                                                   270),
                                          "Accepted angles are 90, 180 and "
                                          "270.")
                                },
                           "sharpen_image":
                               {"filter_strength": ("float",
                                                    lambda filter_strength:
                                                    0<=filter_strength<=1 and
                                                    filter_strength % 2 != 0,
                                                    "Filter strength should be"
                                                    " a float between 0 and 1."
                                                    ),
                               "kernel": ("str",
                                          lambda kernel: kernel in (
                                              "sharpen", "unsharp_mask"),
                                          "Supported kernels are: sharpen, "
                                          "unsharp_mask.")
                                },
                           "stress":
                               {"convert_to_grayscale": ("bool",
                                                         lambda convert_to_grayscale:
                                                         convert_to_grayscale in
                                                         (True, False),
                                                         "Convert_to_grayscale"
                                                         " takes True or "
                                                         "False as input.",),
                                "gamma": ("float",
                                          lambda gamma: 0<=gamma,
                                          "Gamma should be non-integer number "
                                          "equal to or greater than 0."),
                                "iterations": ("int",
                                               lambda iters: 0<iters,
                                               "Iterations should be a "
                                               "positive integer."),
                                "radius": ("int",
                                           lambda radius: 0<radius,
                                           "Radius should be a positive "
                                           "integer."),
                                "sampling": ("int",
                                             lambda sampling: 0 < sampling,
                                             "Sampling should be a positive "
                                             "integer.")
                                },
                           "thresholding":
                               {"adaptiveMethod": ("int",
                                                   lambda adaptiveMethod:
                                                   adaptiveMethod in (0, 1),
                                                   "adaptiveMethod takes "
                                                   "following values: \n 0 "
                                                   "for mean thresholding \n 1"
                                                   " for gaussian thresholding"
                                                   ),
                                "blockSize": ("int",
                                              lambda blockSize:
                                               0<blockSize and
                                               blockSize % 2 != 0,
                                              "blockSize should be a "
                                              "positive odd integer."),
                                "C": ("int",
                                      lambda C: 0<C,
                                      "C should be a positive integer."),
                                "maxValue": ("int",
                                             lambda maxValue:
                                              0<=maxValue<=255,
                                             "maxValue should be between 0 "
                                             "and 255."),
                                "thresholdType": ("str",
                                                  lambda thresholdType:
                                                   thresholdType in (
                                                       "cv2.THRESH_BINARY",
                                                       "cv2.THRESH_BINARY_INV"
                                                   ),
                                                  "thresholdType must be "
                                                  "either cv2.THRESH_BINARY "
                                                  "or cv2.THRESH_BINARY_INV "
                                                  "for reversing black and "
                                                  "white regions.")
                                },
                           },
             "write_output": {"write_image":
                                  {"file_extension": ("str",
                                                     lambda file_extension:
                                                          file_extension.lower()
                                                          in ("jpg", "jpeg",
                                                              "png"),
                                                     "Unsupported file "
                                                     "format. Choose jpg or "
                                                     "png."),
                                  "quality": ("int",
                                              lambda file_extension, quality:
                                                     quality in range(0, 11)
                                               if file_extension == "png" else
                                                     quality in range(1,
                                                                      101),
                                              "Quality should be integer "
                                              "between 0 and 100 for jpeg "
                                              "and between 1 and 10 for png "
                                              "(as compression factor)."),
                              "write_pdf":
                                  {"pdf_compression": ("int",
                                                       lambda
                                                            pdf_compression:
                                                        0 < pdf_compression
                                                        <=10,
                                                       "Pdf compression "
                                                       "value should be an "
                                                       "integer between 1 "
                                                       "and 100."),
                                   "pdf_name": ("str",
                                                lambda pdf_name,
                                                       unpermitted_chars:
                                                unpermitted_chars not in
                                                pdf_name,
                                                "Invalid filename. Check if "
                                                "you are using unpermitted "
                                                "characters."
                                                "")},
                                   }
                              },
             "check_against": ({"unpermitted_chars": ("<", ">", ":", "\"",
                                                      "/", "\\", "|", "?")
                                },)
             }
