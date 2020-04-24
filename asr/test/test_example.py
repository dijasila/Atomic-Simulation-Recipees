import pytest

@pytest.fixture()
def some_input_data():
    return 1


def test_adding_numbers(some_input_data):
    b = 2
    assert some_input_data + b == 3
