import numpy as np 


# add control channels to a sequence
def add_ctrl(seq, ctrl, pos): return np.insert(seq, pos, ctrl, axis=-1)


# create augmented sequence as well as end marker and a dummy sequence
def augment(seq, markers, ctrl_end=None, add_marker=False):
    ctrl_data, ctrl_dummy, pos = markers

    w = add_ctrl(seq, ctrl_data, pos)
    end = add_ctrl(np.zeros((seq.shape[0], 1, seq.shape[2])), ctrl_end, pos)
    if add_marker:
        w = np.concatenate((w, end), axis=1)
    dummy = add_ctrl(np.zeros_like(seq), ctrl_dummy, pos)

    return [w, dummy]