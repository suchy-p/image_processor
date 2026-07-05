import tomllib

from image_processing_pipeline import ImageProcessingPipeline


def load_settings(settings_file: str) -> dict:
    with open(settings_file, "rb") as f:
        return tomllib.load(f)


if __name__ == "__main__":
    settings = load_settings("settings.toml")

    image_processing_pipeline = ImageProcessingPipeline(settings=settings)
    image_processing_pipeline.run_selected_processes()


# todo: rename ImageProcessingPipeline to MainPipeline -> it would be
#  more appropriate since it handles not only image processing, but also
#  file writing and (in future) file conversions
#  // consider renaming ImageProcessor to ImageManipulations or something
#  like that

# todo: move writing files to separate module
#  remove file_counter form ImageProcessor afterwards

# todo: add sanitization layer for settings.toml -> you have some
#  sanitizations in ImageProcessor methods, move them there, create
#  whitelist for accepted settings and values

# todo: use input_dir and output_dir, drop cwd usage <-- remove from
#  write_pdf when moving it to separate module, besides done



# todo: update image_processing_pipeline for removing checker (renamed to
#  check_defaults overwrite) <-- done

# todo: refactor stress for removing checker <-- done

# todo: move all functions' args except settings to a constant since they
#  serve as defaults <-- done

# todo: test all processes <-- done

# todo: update docstrings of ImageProcessor methods <-- done

# todo: update settings.toml to use opencv2 args names <-- done

# todo: move stress to main dir <-- done

# todo: check unsharp masking <-- done

# todo: check ImageProcessor's class args for duplicating those from
#  ImageProcessingPipeline <-- done