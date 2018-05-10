from misc.app_state import AppState
import numpy as np


class ModelBase(object):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.app_state = AppState()

        if self.app_state.visualize:
            from misc.time_plot import TimePlot
            self.plot = TimePlot()

    def plot_sequence(self, input_seq, output_seq, target_seq):
        from matplotlib.figure import Figure
        # Test code

        figs = []

        input_seq = input_seq.numpy()
        output_seq = output_seq.numpy()
        target_seq = target_seq.numpy()

        x = np.zeros(input_seq.shape)
        y = np.zeros(output_seq.shape)
        z = np.zeros(target_seq.shape)

        for i, (input_word, output_word, target_word) in enumerate(zip(input_seq, output_seq, target_seq)):
            fig = Figure()
            axes = fig.subplots(1, 3, sharey=True, sharex=False,
                                gridspec_kw={'width_ratios': [input_seq.shape[1],
                                                              output_seq.shape[1],
                                                              target_seq.shape[1]]})

            x[i] = input_word
            y[i] = output_word
            z[i] = target_word
            axes[0].imshow(x, aspect="equal")
            axes[0].set_title("inputs")
            axes[1].imshow(y, aspect="equal")
            axes[1].set_title("outputs")
            axes[2].imshow(z, aspect="equal")
            axes[2].set_title("targets")
            figs.append(fig)

        self.plot.update(figs)
        return self.plot.is_closed
