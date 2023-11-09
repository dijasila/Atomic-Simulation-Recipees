"""
This is an example file for implementing tests for all Result children classes.

For data integrity and consistency in the UI, you must ensure that each of
these methods is implemented and produces the expected results.
"""
import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from ase.db.row import AtomsRow


data = {'name': 'John', 'age': 30, 'city': 'New York', 'country': 'USA'}


@dataclass
class Result:
    name: str
    age: int
    city: str
    country: str

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_json(cls, string: str) -> object:
        """
        Recreate the object from a string specifying the path to a file or a
        string representation of a json file. E.g., json_str=json.dumps(obj).
        """
        try:
            dct = json.loads(string)
        except:
            with open(string, "r") as file:
                dct = json.load(file)
        return cls(**dct)

    def to_json(self, filename: [str, bool] = False):
        json_str = json.dumps(self.__dict__)

        if filename:
            with open(filename, "w") as file:
                file.write(json_str)
        else:
            return json_str

    def to_pandas(self, keys: list = []):
        keys = self.__dict__ if len(keys) == 0 else keys

        filtered_data = {key: self[key] for key in keys}

        df = pd.DataFrame.from_dict(filtered_data, orient='index')

        return df

    def __getitem__(self, item):
        return self.__dict__[item]


class MissingDataError(Exception):
    pass


def _test_tofrom_dict(data):
    your_instance = Result(**data)

    # dump the object to a dictionary
    inst_dct = your_instance.to_dict()
    assert isinstance(inst_dct, dict)
    assert data == inst_dct

    # objects created from a dict representation of an object are equal to
    # the original object
    inst = Result.from_dict(inst_dct)
    assert inst == your_instance


def _test_tofrom_json(data):
    your_instance = Result(**data)

    # to_json dumps to a string type: if no file is give
    inst_json = your_instance.to_json()
    assert isinstance(inst_json, str)

    # we can recreate the object from a loaded json string
    inst_from_json = your_instance.from_json(inst_json)
    assert isinstance(inst_from_json, Result)

    # if filename is given, to_json dumps to a file return None
    json_file = 'test_json.json'
    inst_json = your_instance.to_json(filename=json_file)

    assert inst_json is None  # writing to a file, logically, nothing is returned

    json_file = Path(json_file)  # check that the json file is created
    assert json_file.is_file() and json_file.stat().st_size != 0


def _test_topandas(data):
    # Example usage:
    your_instance = Result(**data)

    # Call the to_pandas method with no key
    print('passing nothing to pandas df')
    result_df = your_instance.to_pandas()
    print(result_df, '\n')

    # Call the to_pandas method with a single key
    print('passing one key to pandas df')
    result_df = your_instance.to_pandas(['name'])
    print(result_df, '\n')

    # Call the to_pandas method with multiple keys as a set
    print('passing many keys to pandas df')
    result_df = your_instance.to_pandas(['name', 'age', 'country'])
    print(result_df)
