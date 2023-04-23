from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import bootstrap
from collections import Counter


# read xtck output

class xtck_hybrid:
    def __init__(self, hybrid_out, start=-0.0001, end=4e7):
        """
        :param hybrid_out: output from xtck
        :param start: time of the first frame that will be read (ps)
        :param end: time of the last frame that will be read (ps)
        """
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
                        confidence_level=confidence, vectorized=False, axis=0, method=method)
        confidence_dict = {}
        for i in range(len(self.occ_keys)):
            k = self.occ_keys[i]
            confidence_dict[k] = [res.confidence_interval.low[i],
                                  res.confidence_interval.high[i],
                                  ]
        return res, confidence_dict

    def get_state_list(self, state_num=4):
        state_list = []
        if state_num == 4:
            df = self.get_state_df()
            for frame in df.iterrows():
                s = frame[1]
                state_list.append(s["S1"] + s["S2"] + s["S3"] + s["S4"])
        elif state_num == 6:
            df = self.get_state_df()
            for frame in df.iterrows():
                s = frame[1]
                state_list.append(s["S0"] + s["S1"] + s["S2"] + s["S3"] + s["S4"] + s["S5"])
        else:
            raise ValueError("state_num can only be 4 or 6, but "+str(state_num)+" is given.")
        return state_list


    def set_state_set(self, state_num):
        state_list = self.get_state_list(state_num)
        count_dict = dict(Counter(state_list))
        state_set = set(count_dict)
        self.state_set_list = list(state_set)

    def get_state_distribution(self, state_list):
        """
        :param state_list:
        :return: a dictionary with the percentage of occurrences of each element in state_list
        """
        count_dict = dict(Counter(state_list))
        percentage_dict = {k: 0 for k in self.state_set_list}
        for k in count_dict:
            percentage_dict[k] = count_dict[k] / len(state_list)
        percentage_list = [percentage_dict[k] for k in self.state_set_list]
        return percentage_list

    def get_state_distri(self, state_num=4):
        """
        :return: a dictionary with the percentage of occurrences of each element in state_list
        """
        state_list = self.get_state_list(state_num)
        count_dict = dict(Counter(state_list))
        for state in count_dict:
            count_dict[state] /= len(state_list)
        self.state_set = set(count_dict)
        return count_dict

    def get_state_distribution_bootstrap_frame(self, state_num, n_resamples=1000, confidence=0.95, method='BCa'):

        self.set_state_set(state_num)
        data = (self.get_state_list(state_num),)
        res = bootstrap(data,
                        self.get_state_distribution,
                        n_resamples=n_resamples,
                        confidence_level=confidence, vectorized=False, axis=0, method=method)
        return res




class HREX_result_hybrid:
    def __init__(self, result_list, start=-0.0001, end=4e7):
        self.xtck_list = []
        for f in result_list:
            self.xtck_list.append(xtck_hybrid(f, start, end))

    def get_state_df(self):
        state_df_list = []
        for xtck in self.xtck_list:
            state_df_list.append(xtck.get_state_df())
        return state_df_list

    def get_occupancy_bootstrap_frame(self, n_resamples=1000, confidence=0.95, method='BCa'):
        keys = self.xtck_list[0].occ_keys
        result_dict = {k: [[], [],[]] for k in keys}
        for xtck in self.xtck_list:
            _, c_dict_tmp = xtck.get_occupancy_bootstrap_frame(n_resamples=n_resamples, confidence=confidence, method=method)
            occ_dict = xtck.get_occupancy()
            for k in keys:
                result_dict[k][0].append(occ_dict[k])       # mean
                result_dict[k][1].append(c_dict_tmp[k][0])  # lower boundary
                result_dict[k][2].append(c_dict_tmp[k][1])  # higher boundary
        return result_dict

    def get_state_distribution_bootstrap_frame(self, state_num=4, n_resamples=1000, confidence=0.95, method='BCa'):
        state_distribution_all = []
        state_confidence_interval_all = []
        for xtck in self.xtck_list:
            distribution_dict_tmp = xtck.get_state_distri(state_num)
            state_distribution_all.append(distribution_dict_tmp)
            # bootstrap
            res = xtck.get_state_distribution_bootstrap_frame(
                state_num=state_num,
                n_resamples=n_resamples, confidence=confidence, method=method)
            state_confidence_interval_tmp = {}
            for state, low, high in zip(xtck.state_set_list,
                                        res.confidence_interval.low,
                                        res.confidence_interval.high):
                state_confidence_interval_tmp[state] = [distribution_dict_tmp[state],
                                                        low,
                                                        high]
            state_confidence_interval_all.append(state_confidence_interval_tmp)
        # get all the states in all the replicas
        states_set = set()
        for di in state_distribution_all:
            for state in di:
                states_set.add(state)
        # each state has a list of data, which contains [index, occurrence, lower_boundary, upper_boundary ]
        states_result_dict = {}
        for state in states_set:  # loop over states
            states_result_dict[state] = {"index":[], "occurrence":[], "lower":[], "upper":[]}
            for rep_ind in range(len(self.xtck_list)):  # loop over replicas
                di = state_distribution_all[rep_ind]
                conf = state_confidence_interval_all[rep_ind]
                if state in di:
                    states_result_dict[state]["index"].append(rep_ind)
                    states_result_dict[state]["occurrence"].append(di[state])
                    assert di[state] == conf[state][0]
                    states_result_dict[state]["lower"].append(conf[state][1])
                    states_result_dict[state]["upper"].append(conf[state][2])
        return states_result_dict
