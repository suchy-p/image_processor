import tomllib

from image_processing_pipeline import ImageProcessingPipeline


def load_settings(settings_file: str) -> dict:
    with open(settings_file, "rb") as f:
        return tomllib.load(f)


if __name__ == "__main__":
    settings = load_settings("settings.toml")

    image_processing_pipeline = ImageProcessingPipeline(settings=settings)
    image_processing_pipeline.run_selected_processes()
# todo: update image_processing_pipeline for removing checker (renamed to
#  check_defaults overwrite) <-- done
# todo: refactor stress for removing checker
# todo: move all functions' args except settings to a constant since they
#  serve as defaults <-- done except stress
# todo: test all processes <-- done except stress
# todo: update docstrings of ImageProcessor methods <-- done
# todo: check ImageProcessor's class args for duplicating those from
#  ImageProcessingPipeline
# todo: update settings.toml to use opencv2 args names <-- done
# todo: rename ImageProcessingPipeline to MainPipeline -> it would be
#  more appropriate since it handles not only image processing, but also
#  file writing and (in future) file conversions
# todo: move writing files to separate module
# todo: use input_dir and output_dir, drop cwd usage <-- in progress / done
# todo: add sanitization layer for settings.toml -> you have some
#  sanitizations in ImageProcessor methods, move them there, create
#  whitelist for accepted settings and values
# todo: move stress to main dir
# todo: check unsharp masking <-- done