import unicodedata
from os import PathLike
from pathlib import Path
import re
from typing import AnyStr
from zipfile import ZipFile
from zlib import DEFLATED

from pycm import ConfusionMatrix
from pydantic import BaseModel


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def deserialize_from_cm_obj_zip(zip_path: PathLike) -> ConfusionMatrix:
    """
    Deserializes a ConfusionMatrix object from a zip file.

    Args:
        zip_path (PathLike): The path to the zip file containing the PyCM object.

    Returns:
        ConfusionMatrix: The deserialized ConfusionMatrix object.
    """
    with ZipFile(zip_path, "r", compression=DEFLATED) as zipf:
        with zipf.open("pycm.obj", "r") as f:
            cm = ConfusionMatrix(file=f)
    return cm


class File(BaseModel):
    name: str
    path: PathLike
    mode: str = "r"

    @property
    def content(self) -> AnyStr:
        """
        Returns the content of the file as a string.

        Returns:
            str: The content of the file.
        """
        with open(self.path, mode=self.mode) as f:
            return f.read()

    @classmethod
    def from_path(cls, path: PathLike, mode: str = "r") -> "File":
        path = Path(path)
        return cls(name=path.name, path=path, mode=mode)


class Folder(BaseModel):
    name: str
    files: list[File] = []
    folders: list["Folder"] = []

    @classmethod
    def from_path(cls, path: PathLike) -> "Folder":
        """
        Recursively reads the contents of a folder and returns a Folder instance.

        Args:
            path (PathLike): The path to the folder.

        Returns:
            Folder: An instance of Folder with the name, files, and subfolders populated.
        """
        path = Path(path)
        folder = cls(name=path.name)
        for item in path.iterdir():
            if item.is_file():
                folder.files.append(File(name=item.name, path=item))
            elif item.is_dir():
                folder.folders.append(cls.from_path(item))
        return folder


class CSVFile(File):
    @property
    def df(self):
        """
        Returns the content of the CSV file as a pandas DataFrame.

        Returns:
            pd.DataFrame: The DataFrame representation of the CSV content.
        """
        import pandas as pd
        from io import StringIO

        content = self.content
        return pd.read_csv(
            StringIO(content), sep=",", encoding="utf-8", engine="python"
        )


class PYCMObject(File):

    @classmethod
    def from_path(cls, path: PathLike, mode: str = "r") -> "PYCMObject":
        path = Path(path)
        return cls(name=path.name, path=path, mode=mode)

    @property
    def pycm(self):
        """
        Returns the content of the PYCM object file as a ConfusionMatrix instance.

        Returns:
            ConfusionMatrix: The ConfusionMatrix instance created from the file content.
        """
        return deserialize_from_cm_obj_zip(self.path)
