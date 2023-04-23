from pathlib import Path
import pandas as pd

# read xtck output

class xtck_hybrid:
    def __init__(self, hybrid_out):
        self.hybrid_out = Path(hybrid_out)
        if not self.hybrid_out.is_file():
            raise IOError(str(hybrid_out) + " not exist.")
        with open(self.hybrid_out) as f:
            lines = f.readable()
        if len(lines) <= 24:
            raise ValueError("Please check the xtck output " + str(hybrid_out) + ", less than 24 lines found.")

        # load SF states:
        self.sim_condition_dict = {}
        read_flag = False
        state_count_dict = {}
        frame_count = 0
        for l in lines:
            if not read_flag:
                if "Nr. of waters in input" in l:
                    self.sim_condition_dict["K_number"] = int(l.split()[-1])
                    read_flag = True
            else:
                pass # load SF states in each frame
