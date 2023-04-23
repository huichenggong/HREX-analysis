from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import bootstrap


# read xtck output

class xtck_hybrid:
    def __init__(self, hybrid_out, start=-0.0001, end=4e7):
        self.hybrid_out = Path(hybrid_out)
        if not self.hybrid_out.is_file():
            raise IOError(str(hybrid_out) + " not exist.")
        with open(self.hybrid_out) as f:
            lines = f.readlines()
        if len(lines) <= 24:
            raise ValueError("Please check the xtck output " + str(hybrid_out) + ", less than 24 lines found.")

        # load SF states:
        self.sim_condition_dict = {}
        read_flag = False
        state_count_dict = {
            "Time": [],
            "S0": [],
            "S1": [],
            "S2": [],
            "S3": [],
            "S4": [],
            "S5": []}
        frame_count = 0
        for l in lines:
            if not read_flag:
                if "Nr. of waters in input" in l:
                    self.sim_condition_dict["wat_number"] = int(l.split()[-1])
                    read_flag = True
                elif "Nr. of K ions in input" in l:
                    self.sim_condition_dict["K_number"] = int(l.split()[-1])
            else:
                if "Nr. of K+ permeation up" in l:
                    read_flag = False
                    break
                words = l.split()
                time = float(words[0])
                if time >= start and time <= end:
                    state_count_dict["Time"].append(float(words[0]))
                    state = words[4]
                    state_count_dict["S0"].append(state[0])
                    state_count_dict["S1"].append(state[1])
                    state_count_dict["S2"].append(state[2])
                    state_count_dict["S3"].append(state[3])
                    state_count_dict["S4"].append(state[4])
                    state_count_dict["S5"].append(state[5])
                    frame_count += 1
        self.sim_condition_dict["frame_number"] = frame_count
        self.state_df = pd.DataFrame(state_count_dict)

        self.occ_keys = []
        for site in ["S0", "S1", "S2", "S3", "S4", "S5"]:
            for element in ["K", "0", "W"]:
                self.occ_keys.append(site + "_" + element)

    def get_state_df(self):
        return self.state_df

    def get_occu(self, df):
        occup_dict = {}
        for site in ["S0", "S1", "S2", "S3", "S4", "S5"]:
            for element in ["K", "0", "W"]:
                occup_dict[site + "_" + element] = np.mean(df[site] == element)
        return occup_dict

    def get_occupancy(self):
        return self.get_occu(self.state_df)

    def get_occu_numpy(self, d0, d1, d2, d3, d4, d5, ):
        occ = []
        for k, data in zip(self.occ_keys, [d0, d0, d0,
                                           d1, d1, d1,
                                           d2, d2, d2,
                                           d3, d3, d3,
                                           d4, d4, d4,
                                           d5, d5, d5, ]):
            occ.append(np.mean(data == k[-1]))
        return occ

    def get_occupancy_bootstrap_frame(self, n_resamples=1000, confidence=0.95, method='BCa'):
        """
        :param n_resamples: The number of resamples performed to form the bootstrap distribution of the statistic.
        :param confidence: The confidence level of the confidence interval.
        :return:
            BootstrapResult :An object with attributes: confidence_interval, bootstrap_distribution, standard_error
            confidence_dict :A dictionary which contain the lower and higher boundary of the confidence interval
        """
        data = (self.state_df["S0"],
                self.state_df["S1"],
                self.state_df["S2"],
                self.state_df["S3"],
                self.state_df["S4"],
                self.state_df["S5"],)
        res = bootstrap(data,
                        self.get_occu_numpy,
                        n_resamples=n_resamples,
                        paired=True,
                        confidence_level=confidence, vectorized=False, axis=0, method=method )
        confidence_dict = {}
        for i in range(len(self.occ_keys)):
            k = self.occ_keys[i]
            confidence_dict[k] = [res.confidence_interval.low[i],
                                  res.confidence_interval.high[i],
                                  ]
        return res, confidence_dict


class HREX_result_hybrid:
    def __init__(self, result_list):
        self.xtck_list = []
        for f in result_list:
            self.xtck_list.append(xtck_hybrid(f))

    def get_state_df(self):
        state_df_list = []
        for xtck in self.xtck_list:
            state_df_list.append(xtck.get_state_df())
        return state_df_list

