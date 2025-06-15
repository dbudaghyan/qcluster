import unittest
import tempfile
import shutil
from pathlib import Path
import zipfile
from zlib import DEFLATED

from pycm import ConfusionMatrix
import pandas as pd

from qcluster.datamodels.filesystem import (
    slugify,
    deserialize_from_cm_obj_zip,
    File,
    Folder,
    CSVFile,
    PYCMObject,
)


class TestFilesystem(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_slugify(self):
        self.assertEqual(slugify("Hello World"), "hello-world")
        self.assertEqual(slugify("  leading and trailing  "), "leading-and-trailing")
        self.assertEqual(slugify("Special!@#$%^&*()_+-=[]{}|;':\",./<>?`~"), "special")
        self.assertEqual(slugify("你好世界", allow_unicode=True), "你好世界")
        self.assertEqual(slugify("你好世界"), "")
        self.assertEqual(slugify("  --multiple--dashes--  "), "multiple-dashes")

    def test_file(self):
        file_path = self.test_dir / "test.txt"
        file_path.write_text("hello")

        # Test File.from_path
        file_obj = File.from_path(file_path)
        self.assertEqual(file_obj.name, "test.txt")
        self.assertEqual(file_obj.path, file_path)

        # Test content property
        self.assertEqual(file_obj.content, "hello")

    def test_folder(self):
        # Create a directory structure
        (self.test_dir / "subfolder").mkdir()
        (self.test_dir / "file1.txt").touch()
        (self.test_dir / "subfolder" / "file2.txt").touch()

        folder_obj = Folder.from_path(self.test_dir)
        self.assertEqual(folder_obj.name, self.test_dir.name)
        self.assertEqual(len(folder_obj.files), 1)
        self.assertEqual(folder_obj.files[0].name, "file1.txt")
        self.assertEqual(len(folder_obj.folders), 1)
        self.assertEqual(folder_obj.folders[0].name, "subfolder")
        self.assertEqual(len(folder_obj.folders[0].files), 1)
        self.assertEqual(folder_obj.folders[0].files[0].name, "file2.txt")

    def test_csv_file(self):
        csv_path = self.test_dir / "test.csv"
        csv_content = "a,b,c\n1,2,3"
        csv_path.write_text(csv_content)

        csv_file_obj = CSVFile.from_path(csv_path)
        df = csv_file_obj.df
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (1, 3))
        self.assertEqual(list(df.columns), ["a", "b", "c"])
        self.assertEqual(df.iloc[0].tolist(), [1, 2, 3])

    def _create_dummy_cm_zip(self, zip_path):
        cm = ConfusionMatrix(actual_vector=[1, 1, 2, 2], predict_vector=[1, 2, 1, 2])
        obj_path = self.test_dir / "pycm.obj"
        obj_path_for_cm = self.test_dir / "pycm"
        cm.save_obj(obj_path_for_cm.as_posix())

        with zipfile.ZipFile(zip_path, "w", compression=DEFLATED) as zipf:
            zipf.write(obj_path, "pycm.obj")

        obj_path.unlink()  # remove the intermediate file

    def test_deserialize_from_cm_obj_zip(self):
        zip_path = self.test_dir / "test.zip"
        self._create_dummy_cm_zip(zip_path)

        cm = deserialize_from_cm_obj_zip(zip_path)
        self.assertIsInstance(cm, ConfusionMatrix)
        self.assertEqual(cm.actual_vector, [1, 1, 2, 2])
        self.assertEqual(cm.predict_vector, [1, 2, 1, 2])

    def test_pycm_object(self):
        zip_path = self.test_dir / "test.zip"
        self._create_dummy_cm_zip(zip_path)

        pycm_obj = PYCMObject.from_path(zip_path)
        cm = pycm_obj.pycm
        self.assertIsInstance(cm, ConfusionMatrix)
        self.assertEqual(cm.actual_vector, [1, 1, 2, 2])
        self.assertEqual(cm.predict_vector, [1, 2, 1, 2])


if __name__ == "__main__":
    unittest.main()
