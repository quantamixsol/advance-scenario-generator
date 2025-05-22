import unittest
import pandas as pd
from pricing import _tenor_to_years, get_pricing_function

class PricingTests(unittest.TestCase):
    def test_tenor_to_years(self):
        self.assertAlmostEqual(_tenor_to_years('6M'), 0.5)
        self.assertAlmostEqual(_tenor_to_years('5Y'), 5.0)
        self.assertEqual(_tenor_to_years('BAD'), 0.0)

    def test_bond_pnl_with_dv01(self):
        fn = get_pricing_function('BOND')
        row = pd.Series({'quantity': 10, 'dv01': 0.5, 'shock_pct': 25})
        pnl = fn(row)
        self.assertAlmostEqual(pnl, -10 * 0.5 * 25)

    def test_equity_delta(self):
        fn = get_pricing_function('EQ')
        row = pd.Series({'quantity': 100, 'price': 20, 'delta':0.5, 'shock_pct': 10})
        pnl = fn(row)
        self.assertAlmostEqual(pnl, -100*20*0.5*0.10)

if __name__ == '__main__':
    unittest.main()
