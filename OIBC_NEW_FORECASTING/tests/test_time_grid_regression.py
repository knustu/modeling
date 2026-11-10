"""시간대 표현과 잘못된 시간축을 구분하는 합성 회귀 테스트."""
import unittest
import pandas as pd
from src.core import ContractError,official_grid,validate_forecast


class TimeGridRegressionTests(unittest.TestCase):
    def frame(self):
        grid=official_grid().tz_convert('Asia/Seoul')
        return pd.concat([pd.DataFrame({'group':group,'ts':[t.isoformat() for t in grid],
                                        'forecast_kwh':0.0}) for group in ('G1','G2')],ignore_index=True)

    def test_equivalent_kst_grid(self):
        self.assertTrue(validate_forecast(self.frame())['valid_summary_contract'])

    def test_reversed_rows_still_cover_grid(self):
        self.assertTrue(validate_forecast(self.frame().iloc[::-1])['valid_summary_contract'])

    def test_one_hour_shift_rejected(self):
        frame=self.frame()
        frame['ts']=[(pd.Timestamp(v)+pd.Timedelta(hours=1)).isoformat() for v in frame.ts]
        with self.assertRaises(ContractError):validate_forecast(frame)

    def test_duplicate_hour_rejected(self):
        frame=self.frame()
        frame.loc[1,'ts']=frame.loc[0,'ts']
        with self.assertRaises(ContractError):validate_forecast(frame)

    def test_wrong_text_timezone_rejected(self):
        frame=self.frame()
        frame['ts']=[pd.Timestamp(v).tz_convert('UTC').isoformat() for v in frame.ts]
        with self.assertRaises(ContractError):validate_forecast(frame)


if __name__=='__main__':unittest.main()
