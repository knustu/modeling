import unittest
import pandas as pd
from src.contracts import continuous_hourly_grid,same_time_grid,DiagnosticError


class TimeResolutionTests(unittest.TestCase):
    def grid(self):
        return pd.date_range('2025-01-01',periods=48,freq='h',tz='Asia/Seoul')

    def test_hourly_elapsed_time(self):
        self.assertEqual(len(continuous_hourly_grid(self.grid())),48)

    def test_supported_storage_units(self):
        base=self.grid()
        if not hasattr(base,'as_unit'):
            self.assertTrue(same_time_grid(base,base.tz_convert('UTC')))
            return
        for unit in ['s','ms','us','ns']:
            with self.subTest(unit=unit):
                converted=base.as_unit(unit)
                self.assertEqual(len(continuous_hourly_grid(converted)),48)
                self.assertTrue(same_time_grid(converted,base))

    def test_missing_hour_rejected(self):
        with self.assertRaises(DiagnosticError):continuous_hourly_grid(self.grid().delete(12))

    def test_duplicate_hour_rejected(self):
        base=self.grid()
        with self.assertRaises(DiagnosticError):continuous_hourly_grid(base.append(base[:1]))

    def test_subhour_rejected(self):
        with self.assertRaises(DiagnosticError):continuous_hourly_grid(self.grid()+pd.Timedelta(minutes=1))

    def test_offset_shift_not_equal(self):
        self.assertFalse(same_time_grid(self.grid(),self.grid()+pd.Timedelta(hours=1)))


if __name__=='__main__':unittest.main()
