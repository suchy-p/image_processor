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

def draw_random_samples(height: int,
                        width: int,
                        radius: int,
                        sampling: int,
                        coord: tuple[int, int],
                        ) -> tuple[tuple[int, int], ...]:
    """
    Generuje zbiór unikalnych, losowych współrzędnych w określonym promieniu wokół punktu.

    Współrzędne są ograniczone przez wymiary obrazu, co zapobiega wyjściu poza jego krawędzie.

    :param height: Wysokość obrazu (limit dla osi Y).
    :param width: Szerokość obrazu (limit dla osi X).
    :param radius: Promień (połowa boku kwadratu), w którym odbywa się losowanie.
    :param sampling: Liczba próbek do wylosowania.
    :param coord: Współrzędne punktu środkowego (y, x).
    :return: Krotka zawierająca wylosowane pary współrzędnych (y, x).
    """

    height, width = height, width
    radius = radius
    sampling = sampling
    coord = coord

    # Set min and max values for sampling radius for given coord.
    min_height = coord[0] - radius if coord[0] - radius > 0 else 0
    max_height = coord[0] + radius if coord[0] + radius <= height else height
    min_width = coord[1] - radius if coord[1] - radius > 0 else 0
    max_width = coord[1] + radius if coord[1] + radius <= width else width
    # Get random sampling coords.
    height_coords = random.sample(range(min_height, max_height), sampling)
    width_coords = random.sample(range(min_width, max_width), sampling)

    random_samples_coords = tuple(zip(height_coords, width_coords))

    return random_samples_coords

def get_sample_values(image: MatLike,
                      height: int,
                      width: int,
                      channels: int,
                      sampling: int,
                      random_samples_coords: tuple[tuple[int, int], ...]
                      ) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    Pobiera wartości intensywności pikseli dla zestawu przygotowanych współrzędnych próbkowania.

    Funkcja obsługuje zarówno obrazy jednokanałowe (skala szarości), jak i trzykanałowe (RGB).

    :param image: Obraz źródłowy jako macierz NumPy.
    :param height: Docelowa wysokość wynikowej macierzy próbek.
    :param width: Docelowa szerokość wynikowej macierzy próbek.
    :param channels: Liczba kanałów obrazu (1 lub 3).
    :param sampling: Liczba próbek przypadająca na każdy piksel.
    :param random_samples_coords: Zagnieżdżona struktura współrzędnych do pobrania próbek.
    :return: Macierz NumPy (dla 1 kanału) lub krotka trzech macierzy (dla 3 kanałów) z wartościami próbek.
    """

    image = image
    height, width, channels = height, width, channels
    sampling = sampling
    random_samples_coords = random_samples_coords

    grey = []
    red = []
    green = []
    blue = []

    if channels == 1:
        for random_sample_coords_group in random_samples_coords:
            for random_sample in random_sample_coords_group:
                g = image[random_sample]
                grey.append(g)

        grey = np.array(grey).reshape((height, width, sampling))

        return grey

    elif channels == 3:
        for random_sample_coords_group in random_samples_coords:
            for random_sample in random_sample_coords_group:
                r, g, b = image[random_sample]
                red.append(r)
                green.append(g)
                blue.append(b)

        red = np.array(red).reshape((height, width, sampling))
        green = np.array(green).reshape((height, width, sampling))
        blue = np.array(blue).reshape((height, width, sampling))

        return red, green, blue

def calculate_envelopes(height: int,
                        width: int,
                        channels: int,
                        sample_values: np.ndarray | tuple[np.ndarray, ...]
                        ) -> list[tuple[np.ndarray, ...]]:
    """
    Oblicza lokalne wartości minimalne i maksymalne (obwiednie) dla każdego piksela.

    Przetwarza zebrane próbki, aby wyznaczyć zakres dynamiki (min/max) w lokalnym otoczeniu piksela.

    :param height: Wysokość obrazu.
    :param width: Szerokość obrazu.
    :param channels: Liczba kanałów koloru.
    :param sample_values: Wartości próbek uzyskane z funkcji get_sample_values.
    :return: Lista zawierająca pary (minimum, maksimum) dla każdego piksela i kanału.
    """
    height, width, channels = height, width, channels
    sample_values = sample_values

    envelopes = []

    if channels == 1:
        grey = sample_values
        for h in range(height):
            for w in range(width):
                g = grey[(h, w)]
                local_min_and_max = (np.min(g), np.max(g))
                envelopes.append(local_min_and_max)

    elif channels == 3:
        red, green, blue = sample_values

        for h in range(height):
            for w in range(width):
                r, g, b = red[(h, w)], green[(h, w)], blue[(h, w)]

                min_sample_value = np.array([np.min(r),
                                          np.min(g),
                                          np.min(b)])
                max_sample_value = np.array([np.max(r),
                                          np.max(g),
                                          np.max(b)])
                local_min_and_max = zip(min_sample_value,
                                           max_sample_value)
                envelopes.append(local_min_and_max)

    return envelopes


