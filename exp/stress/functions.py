from os import chdir, getcwd, listdir, path
import random

from cv2.typing import MatLike
import numpy as np

def get_file_list(input_dir: str) -> list[str]:
    """
    Pobiera listę nazw plików graficznych (.jpg, .png) z podanego katalogu.

    Funkcja zmienia bieżący katalog roboczy procesu na `input_dir`, a następnie
    skanuje go w poszukiwaniu plików o rozszerzeniach JPEG i PNG.

    :param input_dir: Ścieżka do katalogu zawierającego obrazy do
    przetworzenia.
    :return: Lista nazw plików pasujących do rozszerzeń .jpg lub .png.
    :raises OSError: Jeśli ścieżka `input_dir` jest nieprawidłowa lub
    niedostępna.
    """

    input_dir = input_dir
    file_list: list[str] = [file for file in listdir(input_dir) if
                            path.isfile(file) and file.endswith((".jpg",
                                                                     ".png"))]

    return file_list


