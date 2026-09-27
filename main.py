import tomllib

from image_processing_pipeline import ImageProcessingPipeline


def load_settings(settings_file: str) -> dict:
    with open(settings_file, "rb") as f:
        return tomllib.load(f)


if __name__ == "__main__":
    settings = load_settings("settings.toml")

    image_processing_pipeline = ImageProcessingPipeline(settings=settings)
    image_processing_pipeline.run_selected_processes()
