import copy
import csv
from pathlib import Path
import unittest

from reproduce_main_table import compute, load_rows


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=load_rows(Path(__file__).resolve().parents[1]/'tables/query_errors')

    def test_paper_values(self):
        table,checks=compute(self.rows)
        self.assertEqual(len(table),6)
        self.assertTrue(all(row['passed'] for row in checks))

    def test_missing_query_rejected(self):
        with self.assertRaises(ValueError):
            compute(self.rows[:-1])

    def test_duplicate_query_rejected(self):
        rows=list(self.rows)
        rows[-1]=rows[0]
        with self.assertRaises(ValueError):
            compute(rows)

    def test_modified_errors_fail_paper_check(self):
        rows=copy.deepcopy(self.rows)
        for row in rows:
            row['cmcgc']='10000'
        with self.assertRaises(ValueError):
            compute(rows)


if __name__=='__main__':
    unittest.main()
