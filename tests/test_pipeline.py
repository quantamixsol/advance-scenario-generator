import unittest
import pandas as pd
from exposures import apply_shocks, validate_parallel_shocks

class PipelineTests(unittest.TestCase):
    def test_apply_shocks(self):
        pf = pd.DataFrame({'ticker':['T1'], 'quantity':[10], 'price':[100], 'rf_code':['RF1']})
        rf = pd.DataFrame({'original':['RF1'], 'asset':['EQ'], 'shock_pct':[10]})
        out = apply_shocks(pf, rf)
        self.assertIn('pnl', out.columns)
        self.assertAlmostEqual(out.pnl.iloc[0], -10*100*0.10)

    def test_validate_parallel(self):
        rf = pd.DataFrame({
            'curve_name':['C1','C1'],
            'shock_pct':[5,5]
        })
        # should not raise
        validate_parallel_shocks(rf)
        rf_bad = pd.DataFrame({'curve_name':['C1','C1'], 'shock_pct':[5,10]})
        with self.assertRaises(ValueError):
            validate_parallel_shocks(rf_bad)

if __name__ == '__main__':
    unittest.main()
