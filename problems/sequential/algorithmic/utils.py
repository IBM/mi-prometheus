import numpy as np 


# add control channels to a sequence
def add_ctrl(seq, ctrl, pos): return np.insert(seq, pos, ctrl, axis=-1)


# create augmented sequence as well as end marker and a dummy sequence
def augment(seq, markers, ctrl_start=None, add_marker_data=False, add_marker_dummy=True):
    ctrl_data, ctrl_dummy, pos = markers

    w = add_ctrl(seq, ctrl_data, pos)
    start = add_ctrl(np.zeros((seq.shape[0], 1, seq.shape[2])), ctrl_start, pos)
    if add_marker_data:
        w = np.concatenate((start, w), axis=1)

    start_dummy = add_ctrl(np.zeros((seq.shape[0], 1, seq.shape[2])), ctrl_dummy, pos)
    ctrl_data_select = np.ones(len(ctrl_data))
    dummy = add_ctrl(np.zeros_like(seq), ctrl_data_select, pos)

    if add_marker_dummy:
       dummy = np.concatenate((start_dummy, dummy), axis=1)

    return [w, dummy]