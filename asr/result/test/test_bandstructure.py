import json

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from ase.db.row import AtomsRow
from ase.db import connect
from ase.dft.kpoints import BandPath

db_path = '/home/tara/website_backend/database.db'

def get_atomsrow(data_path):
    row_iter = connect(data_path).select()
    atoms_row = next(row_iter)

    return atoms_row


atoms_row = get_atomsrow(db_path)
for key in atoms_row.data:
    print(key)
    print('\t', atoms_row.data[key].keys())


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


@dataclass
class Metadata:
    description: str
    author: str
    version: int = 0


@dataclass
class BSNOSOCData:
    """
    Dataclass for storing information related to spin-orbit-coupling
    bandstructure calculations.
    """
    energy: float  #
    """
    """

    path: BandPath  #
    """
    """

    energies: np.ndarray  #
    """
    """

    reference: float  #
    """
    """

    efermi: float  #
    """
    """

    sz_mk: float  #
    """
    """

@dataclass
class BSData:
    name: str  # basic bandstructure dataclass
    """
    Representation of a band structure calculation for a typical
    DFT calculation.
    """

    metadata: Metadata  # store information related to the calculation
    """
    Standard metadata dataclass used to record essential information related to
    the 
    """

    path: BandPath  #
    """
    """

    energies: np.ndarray  #
    """
    """

    reference: float  #
    """
    """

    efermi: float  #
    """
    """

    bs_soc: np.ndarray  # A short description of the bs_soc attribute
    """
    Data related to the BS_SOC calculation.
    """

    bs_nosoc: np.ndarray  # A short description of the bs_nosoc attribute
    """
    Data related to the BS_NOSOC calculation.
    """


@dataclass
class BandstructureResult(Result):
    name: str  # bandstructure results
    """
    User-facing dataclass for collecting results related to different types of 
    band structure calculations.
    """

    bs_soc: BSData
    """
    """

    bs_nosoc: BSNOSOCData
    """
    """

    @classmethod
    def from_row(cls, row: AtomsRow, row_key: str = 'results-asr.gs.json') -> object:
        database_data = row.data.get(row_key, None)

        if database_data is None:
            raise MissingDataError(f"Key not found in row: {row_key}")

        return cls(**database_data)

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


def _some_test_case(YourClass):
    try:
        result = YourClass.from_dict(data)
        print(result.name)
        print(result.age)
    except MissingDataError as e:
        print(f"Error: {e}")

    atoms_row = get_atomsrow(db_path)
    print(atoms_row)