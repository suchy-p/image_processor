import tomllib

from image_processor import ImageProcessor


def load_settings(settings_file: str) -> dict:
    with open(settings_file, "rb") as f:
        return tomllib.load(f)

if __name__ == '__main__':
    settings = load_settings("settings.toml")
    
    image_processor = ImageProcessor(settings=settings)
    image_processor.image_processing_pipeline()
