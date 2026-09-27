# Image Processor
## Overview
Image Processor was created for modifying large batches of image files. Idea of such app rose from a need for improving readability of about 1500 low quality microfilm pictures. In this case using image editors such as Gimp was impractical, some sort of automation tool was necessary.
Main idea was to create an app capable of applying common image enhancing techniques to multiple files, while also allowing for the easy addition of new features in the future. This goal was achieved through the use of the pipeline design pattern.
Rigt now common techniques include: brightness and contrast adjustment, bilateral filtering, contrast limited histogram equalization (clahe), denoising, reversing colors, image rotation, sharpening and unsharp masking, adaptive thresholding. They all belong to OpenCV library.
Besides them ready to use is package implementing Spatio-Temporal Retinex-Inspired Envelope with Stohastic Sampling (STRESS) framework for local contrast and color correction. It was added as a part of a way of dealing with image vignette, which was one of image defects Image Processor was created to mitigate. It has many other uses, for example improving readability of faded text. This implementation is based on this article: [Ø. Kolås, I. Farup, A. Rizzi, „Spatio-Temporal Retinex-Inspired Envelope with Stochastic Sampling: A Framework for Spatial Color Algorithms”, The Journal of Imaging Science and Technology, 2011-07, Vol.55 (4)](https://search-library.ucsd.edu/discovery/fulldisplay/cdi_ingenta_journals_ist_jist_2011_00000055_00000004_art00011/01UCS_SDI:UCSD).

## Used technologies
- Python 3.13
- OpenCV 4.11.0.86
- NumPy 2.2.4

## How does it work
Right now settings.toml file acts as a sort of cli. Each functionality, be it setting color mode, input directory or image modification, operates by using key-value pairs where key represents parameter name and value acts as user input. All settings are grouped into categories: color, paths and processes. We shall go through each functionality, explaining how to use them. These instructions are also included in the settings.toml file, so you shouldn’t need to look for them back here.

### colors
Right now here is only one setting.

#### color_mode
Determines mode in which images are opened, allowing for grayscaling at the beginning of the pipeline.
Expected values: color, grayscale 

### paths

#### input_dir
Directory of files for processing.
Expected value type: string representing a directory path

#### output_dir
Name of folder created in input directory where processed images are written.
Expected value type: string valid as a directory name

### processes
This category gathers main functionalities. They all have few things in common. They can be turned on and of by `enabled` key’s value set to `True` or `False`. Rest of the keys represent given functionality arguments. If value of an argument key is set to `None` it means that default argument value will be used. Each argument has comment explaining what it does, what are expected and default values. 
The argument naming format uses camelCase notation because most of the functions are derived from OpenCV, which employs this naming convention. I chose not to map them to snake_case to avoid possible confusion when consulting library’s documentation and simplify existing functionalities’ modifications or adding new features (you don’t have to remember to change or add mappings).

#### adjust_brightness_and_contrast
Globally increase contrast and brightness. In most cases increasing contrast should go with lowering brightness and vice versa.
##### alpha
Contrast value. Value between 0 and 1 lowers the contrast, while value above 1 increases it.
Expected data type: positive float
Default value: 1.0
##### beta
Brightness value. Suggested value between -127 and 127.
Expected datatype: integer
Default value: 0

#### bilateral_filter
Edge preserving filtering for noise reduction.
##### d
Diameter of the pixel neighborhood used for filtering. Higher value decreases computing performance. D is in strict relation with sigmaSpace, where for optimal performance d ≈ 3 * sigmaSpace. If d ≤ 0, d is calculated automatically using this equation. 
Expected data type: integer
Default value: 0
##### sigmaColor
Color deviation value, i.e. how distant shades of given color will be filtered. 
Expected data type: positive integer
Default value: 15
##### sigmaSpace
Filter range, i.e. how neighboring pixels (d argument above) influence decreases with distance from given pixel. SigmaSpace is in strict relation with d, where for optimal performance d ≈ 3 * sigmaSpace. 
Expected data type: positive integer
Default value: 15

#### clahe (Contrast Limited Histogram Equalization)
Local contrast correction. 
##### clipLimit
Threshold for contrast limiting, i.e. contrast values exceeding this value will be clipped.
Expected data type: positive float 
Default value: 40.0
##### tileGridSize
Size of segments upon which image will be divided. These segments have their own local histograms, which will be equalized instead of a global histogram.
Expected data type: tuple of two positive integers
Default value: (8, 8)

#### denoise_filter
Common denoise filter, faster alternative to bilateral filter.
##### searchWindowSize
Size in pixels of the window that is used to compute weighted average for given pixel. In most cases value above 31 doesn’t result in any significant image improvement. Affects performance.
Expected data type: positive odd integer
Default value: 21
##### templateWindowSize
Size in pixels of the template patch used to search for similarities in search window and compute weights, thus should be smaller than searchWindowSize. Lower values are better at preserving details, higher at heavy noise removal.
Expected data type: positive odd integer
Default value: 7
##### h
Filter strength for grayscale images. Higher value means better noise removal at the cost of removing image details.
Expected data type: positive integer
Default value: 30
##### hColor
Filter strength for color images. Higher value means better noise removal at the cost of removing image details and distorting colors.
Expected data type: positive integer
Default value: 10 

#### reverse_colors
Simple image color reversal. Takes no arguments, it’s either enabled or disabled.

#### rotate_image
Rotate image by 90 degrees clockwise.
Expected values: 90, 180, 270

#### sharpen_image
Applies defined sharpen or unsharp mask kernels.
##### kernel
A kernel which is going to be applied: sharpen or unsharp_mask.
Expected data type: string
Default value: sharpen
##### filter_strength
Weight for alpha blending of original and processed images. Correct values are in range between 0.0 and 1.0, where 0.0 returns original image and 1.0 returns fully filtered image.
Expected data type: float in range 0.0 – 1.0
Default value: 0.5

#### stress
Application of Spatio-Temporal Retinex-inspired Envelope with Stochastic Sampling (STRESS) algorithm. It’s useful for color enhancement, contrast correction and creating grayscale images. The latter is accomplished here by applying STRESS to already grayscaled image. Direct color to grayscale conversion using STRESS is also possible, but requires different computations than used here.
More information and use cases can be found [here.]( https://search-library.ucsd.edu/discovery/fulldisplay/cdi_ingenta_journals_ist_jist_2011_00000055_00000004_art00011/01UCS_SDI:UCSD)
For now it’s only example of added custom feature.
##### convert_to_grayscale
Convert image to grayscale before applying STRESS.
Expected data type: boolean
Default value: false
##### radius
STRESS works in local scope, so it needs to have defined maximum possible sampling distance for each image pixel. Too low radius results in artifacts.
Default radius value varies. According to authors’ recommendation (see link above) it is equal to diagonal of an image in pixels. 
Expected data type: positive integer
Default value: image diagonal in pixels
##### sampling
Number of random samples taken within a specified radius. It highly affects color and contrast enhancement results.
Expected data type: positive integer
Default value: 30
##### iterations
Specifies how many times the STRESS value is going to be calculated before determining means which are returned as output pixels’ values. A higher number of iterations results in smoother tonal transitions.
Expected data type: positive integer
Default value: 10
##### gamma
Gamma correction factor. 1.0 means that no gamma correction is applied.
Expected data type: positive float
Default value: 1.0

#### thresholding
Creates binary (i.e. black and white) images using adaptive thresholding.
##### adaptiveMethod
Choose between two adaptive thresholding methods represented by 0 and 1: mean [0] or gaussian [1].
Expected data type: 0 or 1 as integer
Default value: 0
##### blockSize
Size of pixel neighborhood used to calculate local threshold value.
Expected data type: positive odd integer
Default value: 199
##### C
Constant value subtracted from the mean or weighted (for gaussian thresholding) sum of neighboring pixels.
Expected data type: positive integer
Default value: 40
##### maxValue
Maximum value that can be assigned to a pixel. 255 equals pure white.
Expected data type: positive integer ≤ 255
Default value: 255 
##### thresholdType
Threshold types used by adaptive thresholding in OpenCV library are cv2.THRESH_BINARY (for bringing bright details on dark background) and cv2.THRESH_BINARY_INV (for bringing dark details on bright background).
Expected data type: 0 for cv2.THRESH_BINARY or 1 for cv2.THRESH_BINARY_INV
Default value: 0

## Customization
Whether you want to add new OpenCV-based image manipulation or your own custom feature, you have to follow few rules.

### ImageProcessor or your own package?
`ImageProcessor` class of `image_processor` module is meant to gather all image manipulations derived from OpenCV. Therefore if you want to implement something already existing in this library, that’s the right spot. Just add it as a new class method.
If you are implementing something more complex, then creating a separate module or package is recommended approach. Consider using pipeline pattern if it’s going to run multiple functions in a sequence. This will likely make the next steps easier.

### Default parameters’ values and user settings
Every functionality contains dictionary of it’s own arguments and their default values which are checked against user settings. If user sets an argument value to `None`, it uses default argument value (except `reverse_color` and `rotate_image` methods of `ImageProcessor` which dont have defaults).
Checks are performed by `check_defaults_overwrite` static method from `ImageProcessor` class. It has to be called within a new `ImageProcessor` method or within a function in separate module / package. Unless you are using custom pipeline for sequential processes. In this case you can check user settings at the beginning of the pipeline and pass results to functions. It will make things a lot easier and less error prone.
`Check_defaults_overwrite` takes two arguments: `user_settings` and `default_settings`, returns dict of parameter: value pairs. 

#### Example 1
```**Method of ImageProcessor class**
class *ImageProcessor*
  […]
  @staticmethod
  def *check_defaults_overwrite*(user_settings, default_settings)
  […]
  def *new_method*(self, …, user_settings):
    default_settings = {“arg_1”: value_1, “arg_2”: value_2}
    checked_settings = self.check_defaults_overwrite(user_settings, default_settings)
    […]
```

#### Example 2
```** Separate module / package**
from *image_processor* import *ImageProcessor*

check_defaults_overwrite = ImageProcessor.check_defaults_overwrite

def *new_function*(…, user_settings):
  default_settings = {“arg_1”: value_1, “arg_2”: value_2}
  checked_settings = check_defaults_overwrite(user_settings, default_settings)
  […]
```

#### Example 3
```** Custom pipeline**

from *image_processor* import *ImageProcessor*

check_defaults_overwrite = ImageProcessor.check_defaults_overwrite

def *custom_pipeline* (…, user_settings):
  default_settings = {“arg_1”: value_1, “arg_2”: value_2}
  checked_settings = check_defaults_overwrite(user_settings, default_settings)
  […]
  function_1(…, checked_settings)
  function_2(…, checked_settings)
```

### Registering new feature
Next step is registering your new method / function / custom pipeline as a part of `ImageProcessingPipeline` (from `image_processing_pipeline` module). To do this you need to add it to `to_process loop` in `run_selected_processes` method. If your feature is a method of `ImageProcessor` class, just write it according to template below, otherwise you need to import it first.

#### Example
```from *your_module* import *function_1* \# If needed.

class *ImageProcessingPipeline*:
  […]
  def run_selected_processes(self):
    […]
    for image in to_process:
        […]
        \# For ImageProcessor class method.
        if enabled_processes[“method_1”][“enabled”]:
          image_object = self.processes.function_1(image_object,
                                            […], 
                                            settings=enabled_processes[“method_1”]
                                            )
         \# For imported function.
         if enabled_processes[“function_1”][“enabled”]:
          image_object = self.processes.function_1(image_object,
                                            […], 
                                            settings=enabled_processes[“function_1”]
                                            )
```
 
> **Note:**
Keep in mind, that function / method / pipeline added this way will run in a loop for all images you are going to process. If you want to do something new with output files, like merging them into single PDF file, it needs to be placed outside `to_process loop`. 


### User settings
What’s left is adding your new functionality and it’s arguments to proper section of settings.toml file. There are two sections you should use: `processes` for image manipulations and `write_output` for handling output files. There are no restrictions on arguments, except one. Remember `enabled` key checked in `enabled_processes[method_1]` and `enabled_processes[function_1]` above? This is the only mandatory argument which serves as an on/off switch and takes boolean values.

#### Example
```[processes.new_function]
enabled = true
arg_1 = value_1
arg_2 = value_2
```

### Summary
So, if you want to add new functionality you need to do 3 things:
- Include default arguments values in your new feature and check them against user settings.
- Register it as a process that could be run.
- Add it to settings.toml file, include `enabled = true` beside it’s arguments to serve as an on/off switch.
That’s all, you’re good to go.


## Planned changes
- Validation layer for user input.
- Modification of `STRESS` package to use HSV color space instead of RGB. In HSV processing color images will depend upon applying STRESS algorithm solely to Volume value instead of three separate color channels in RGB. It should result in significant speed gains. Additionally, as grayscale images also depend on a single color value, there will be no need for channel checking and branching computations depending on it’s result.
- Creating of a new module containing functions working with files: rename, change format (including converting from Apple HEIC). File writing functions will be moved here also. 
- Change `sharpen_image` to `apply_kernel`. `Sharpen_image` uses OpenCV `filter2D` function to apply sharpen or unsharp mask kernel. It might as well serve for applying other kernels added in the future. Thus kernels are going to be moved to separate module.
- Refactor `run_selected_processes` method of `ImageProcessingPipeline` to accept list of processes from another file and run them using a loop. It will reduce boilerplate and make adding new features easier.
