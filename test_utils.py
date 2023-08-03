import hashlib

from utils import load_data
from utils import extract_words, create_bag_of_words


def test_file_read():
    filename = "game.tsv"
    data = load_data(filename)
    assert len(data) == 20

    assert data[0] == "This room has a couch, chairs and TV."


def test_extract_words():
    text = "This room has a couch, chairs and TV."
    assert extract_words(text) == [
        "this",
        "room",
        "has",
        "a",
        "couch,",
        "chairs",
        "and",
        "tv.",
    ]


def test_create_bag_of_words():
    data = load_data("game.tsv")
    bow = create_bag_of_words(data)
    hashed_dict = hashlib.md5(
        str(sorted(bow.items())).encode(), usedforsecurity=False
    ).hexdigest()
    assert hashed_dict == "7227b868f8cfdd65bc8f3042c67efb62"
