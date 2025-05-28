import unittest
from unittest.mock import patch

from modules import shocksim

class ShockSimTests(unittest.TestCase):
    def setUp(self):
        self.risk_factors = [
            {"original": "R1", "asset": "IRSWAP", "curve_name": "C1", "tenor": "1Y"},
            {"original": "R2", "asset": "IRSWAP", "curve_name": "C1", "tenor": "5Y"},
            {"original": "E1", "asset": "EQ"},
        ]

    def test_parallel_shift_enforced(self):
        with patch("modules.shocksim._call_llm", return_value="[10, 20, 0.5]"):
            out = shocksim.simulate_shocks(self.risk_factors, "narr", "Medium", engine="gpt-4")
        self.assertEqual(out, [10.0, 10.0, 0.5])

    def test_fallback_to_baseline(self):
        with patch("modules.shocksim._call_llm", return_value="not json"):
            out = shocksim.simulate_shocks(self.risk_factors, "narr", "Medium", engine="gpt-4")
        baseline = [25, 25, 2.5]
        self.assertEqual(out, baseline)

if __name__ == "__main__":
    unittest.main()
