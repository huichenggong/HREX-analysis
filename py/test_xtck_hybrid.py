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

    def test__init__end(self):
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
                                  xtck.state_df["S5"], )
        keys = []

        occ_dict = xtck.get_occupancy()
        self.assertListEqual(occ, [occ_dict[k] for k in xtck.occ_keys])

    def test_get_occupancy_bootstrap_frame(self):
        xtck = xtck_hybrid("01-NaK2K-charge/0/k_hybrid.out")
        res, confidence_dict = xtck.get_occupancy_bootstrap_frame()
        # print(res.confidence_interval)

    def test_get_state_list(self):
        xtck = xtck_hybrid("01-NaK2K-charge/0/k_test.out")
        state_list = xtck.get_state_list()  # default state number 4
        self.assertListEqual(state_list,
                             ["K0KK", "KK0K", "K0KK", "KK0K", "KK0K",
                              "K0KK", "K0K0", "K0K0", "KK0K", "K0KW", ])
        state_list = xtck.get_state_list(6)
        self.assertListEqual(state_list,
                             ["WK0KKW", "WKK0KW", "WK0KKW", "WKK0KW", "WKK0KW",
                              "WK0KKW", "WK0K0W", "WK0K0W", "WKK0KW", "0K0KWW", ])

    def test_get_state_distribution(self):
        xtck = xtck_hybrid("01-NaK2K-charge/0/k_test.out")
        s_num = 4
        xtck.set_state_set(s_num)
        count_list = xtck.get_state_distribution(xtck.get_state_list(s_num))
        count_dict = {xtck.state_set_list[i]: count_list[i] for i in range(len(count_list))}
        self.assertDictEqual(count_dict, {"K0KK": 0.3, "KK0K": 0.4,
                                          "K0K0": 0.2, "K0KW": 0.1
                                          })

        s_num = 6
        xtck.set_state_set(s_num)
        count_list = xtck.get_state_distribution(xtck.get_state_list(s_num))
        count_dict = {xtck.state_set_list[i]: count_list[i] for i in range(len(count_list))}
        self.assertDictEqual(count_dict, {"WK0KKW": 0.3, "WKK0KW": 0.4,
                                          "WK0K0W": 0.2, "0K0KWW": 0.1
                                          })

    def test_get_state_distri(self):
        xtck = xtck_hybrid("01-NaK2K-charge/0/k_test.out")
        count_dict = xtck.get_state_distri(4)
        self.assertDictEqual(count_dict, {"K0KK": 0.3, "KK0K": 0.4,
                                          "K0K0": 0.2, "K0KW": 0.1})
        count_dict = xtck.get_state_distri(6)
        self.assertDictEqual(count_dict, {"WK0KKW": 0.3, "WKK0KW": 0.4,
                                          "WK0K0W": 0.2, "0K0KWW": 0.1})

    def test_get_state_distribution_bootstrap_frame(self):
        xtck = xtck_hybrid("01-NaK2K-charge/0/k_test.out")
        res = xtck.get_state_distribution_bootstrap_frame(
            state_num=4,
            n_resamples=100, confidence=0.95, method='percentile')
        #print(res)

    def test_HREX_get_occupancy_bootstrap_frame(self):
        file_list = ["02-TRAAK-charge/%d/k_hybrid.out" % i for i in range(36)]
        HREX = HREX_result_hybrid(file_list)
        confidence_dict = HREX.get_occupancy_bootstrap_frame(n_resamples=100, method="basic")

    def test_HREX_get_state_distribution_bootstrap_frame(self):
        file_list = ["01-NaK2K-charge/%d/k_test.out" % i for i in range(2)]
        HREX = HREX_result_hybrid(file_list)
        states_result_dict = HREX.get_state_distribution_bootstrap_frame(6)
        self.assertListEqual(states_result_dict["WK0KKW"]["index"], [0, 1])
        self.assertListEqual(states_result_dict["WK0KKW"]["occurrence"], [0.3, 0.6])

        self.assertListEqual(states_result_dict["WKK0KW"]["index"], [0, 1])
        self.assertListEqual(states_result_dict["WKK0KW"]["occurrence"], [0.4, 0.1])

        self.assertListEqual(states_result_dict["0K0KWW"]["index"], [0])
        self.assertListEqual(states_result_dict["0K0KWW"]["occurrence"], [0.1])




if __name__ == '__main__':
    unittest.main()
