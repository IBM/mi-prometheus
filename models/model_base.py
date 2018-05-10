from misc.app_state import AppState
import numpy as np


class ModelBase(object):
    """ Class representing base class of all models.
    Provides basic plotting functionality.
    """
    def __init__(self):
        """ Initializes application state and sets plot if visualization flag is turned on."""
        super(ModelBase, self).__init__()
        self.app_state = AppState()

        if self.app_state.visualize:
            from misc.time_plot import TimePlot
            self.plot = TimePlot()

    def plot_sequence(self, input_seq, output_seq, target_seq):
        """ Creates a default interactive visualization, with a slider enabling to move forth and back along the time axis (iteration in a given episode).
        The default visualizatoin contains input, output and target sequences.
        For more model/problem dependent visualization please overwrite this method in the derived model class.
        """
        from matplotlib.figure import Figure
        # Test code

        figs = []

        # Change to np arrays and transpose, so x will be time axis.
        print("input_seq  size =", input_seq.size())
        input_seq = (input_seq.numpy())
        output_seq = (output_seq.numpy())
        target_seq = (target_seq.numpy())

        x = np.transpose(np.zeros(input_seq.shape))
        y = np.transpose(np.zeros(output_seq.shape))
        z = np.transpose(np.zeros(target_seq.shape))

        for i, (input_word, output_word, target_word) in enumerate(zip(input_seq, output_seq, target_seq)):
            fig = Figure()
            axes = fig.subplots(3, 1, sharex=True, sharey=False,
                                gridspec_kw={'width_ratios': [input_seq.shape[0]]})

            x[:, i] = input_word
            y[:, i] = output_word
            z[:, i] = target_word
            axes[0].imshow(x, aspect="equal")
            axes[0].set_title("inputs")
            axes[1].imshow(y, aspect="equal")
            axes[1].set_title("outputs")
            axes[2].imshow(z, aspect="equal")
            axes[2].set_title("targets")
            figs.append(fig)

        self.plot.update(figs)
        return self.plot.is_closed
