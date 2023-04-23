import unittest
from xtck_hybrid import *


class MyTestCase(unittest.TestCase):
    def test_sim_condition_dict(self):
        xtck = xtck_hybrid("01-NaK2K-charge/0/k_hybrid.out")
        self.assertEqual(xtck.sim_condition_dict["wat_number"], 11134)
        self.assertEqual(xtck.sim_condition_dict["K_number"], 160)
        self.assertEqual(xtck.sim_condition_dict["frame_number"], 2001)
        self.assertEqual(xtck.state_df.iloc[0]["S0"], "W")
        self.assertEqual(xtck.state_df.iloc[0]["S1"], "K")
        self.assertEqual(xtck.state_df.iloc[0]["S2"], "0")
        self.assertEqual(xtck.state_df.iloc[0]["S3"], "K")
        self.assertEqual(xtck.state_df.iloc[0]["S4"], "K")
        self.assertEqual(xtck.state_df.iloc[0]["S5"], "W")
        df = xtck.get_state_df()
        self.assertEqual(df.iloc[1]["S0"], "W")
        self.assertEqual(df.iloc[1]["S1"], "K")
        self.assertEqual(df.iloc[1]["S2"], "K")
        self.assertEqual(df.iloc[1]["S3"], "0")
        self.assertEqual(df.iloc[1]["S4"], "K")
        self.assertEqual(df.iloc[1]["S5"], "W")

    def test__init__(self):
        xtck = xtck_hybrid("01-NaK2K-charge/0/k_hybrid.out", end=20000.1)
        self.assertEqual(xtck.sim_condition_dict["wat_number"], 11134)
        self.assertEqual(xtck.sim_condition_dict["K_number"], 160)
        self.assertEqual(xtck.sim_condition_dict["frame_number"], 1001)
    def test_get_occupancy(self):
        xtck = xtck_hybrid("01-NaK2K-charge/0/k_test.out")
        self.assertEqual(xtck.sim_condition_dict["frame_number"], 10)
        occ_dict = xtck.get_occupancy()
        self.assertAlmostEqual(occ_dict["S0_K"], 0.0)
        self.assertAlmostEqual(occ_dict["S0_0"], 0.1)
        self.assertAlmostEqual(occ_dict["S0_W"], 0.9)
        self.assertAlmostEqual(occ_dict["S1_K"], 1)

        self.assertAlmostEqual(occ_dict["S2_K"], 0.4)
        self.assertAlmostEqual(occ_dict["S2_0"], 0.6)

        self.assertAlmostEqual(occ_dict["S4_K"], 0.7)
        self.assertAlmostEqual(occ_dict["S4_W"], 0.1)
        self.assertAlmostEqual(occ_dict["S4_0"], 0.2)

        self.assertAlmostEqual(occ_dict["S5_K"], 0.0)
        self.assertAlmostEqual(occ_dict["S5_W"], 1)
        self.assertAlmostEqual(occ_dict["S5_0"], 0.0)

    def test_get_occu_numpy(self):
        xtck = xtck_hybrid("01-NaK2K-charge/0/k_test.out")
        occ = xtck.get_occu_numpy(xtck.state_df["S0"],
                                  xtck.state_df["S1"],
                                  xtck.state_df["S2"],
                                  xtck.state_df["S3"],
                                  xtck.state_df["S4"],
                                  xtck.state_df["S5"],)
        keys = []

        occ_dict = xtck.get_occupancy()
        self.assertListEqual(occ, [occ_dict[k] for k in xtck.occ_keys])

    def test_get_occupancy_bootstrap_frame(self):
        xtck = xtck_hybrid("01-NaK2K-charge/0/k_hybrid.out")
        res, confidence_dict = xtck.get_occupancy_bootstrap_frame()
        #print(res.confidence_interval)






if __name__ == '__main__':
    unittest.main()
