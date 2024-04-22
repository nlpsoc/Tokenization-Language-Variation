from unittest import TestCase
from styletokenizer.utility.filesystem import get_dir_to_src


class FilesystemTest(TestCase):
    def test_get_dir_to_src(self):
        src_path = get_dir_to_src()
        self.assertEqual(src_path, "/Users/anna/Documents/git projects.nosync/StyleTokenizer/src")
