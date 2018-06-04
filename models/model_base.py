from misc.app_state import AppState
import numpy as np
import logging

class ModelBase(object):
    """ Class representing base class of all models.
    Provides basic plotting functionality.
    """
    def __init__(self):
        """ Initializes application state and sets plot if visualization flag is turned on."""
        super(ModelBase, self).__init__()
        # WARNING: at that momen AppState must be initialized and flag must be set. Otherwise the object plot won't be created.
        # SOLUTION: if application is supposed to show dynamic plot, set flag to True before constructing model! (and set to False right after if required)
        self.app_state = AppState()

        if self.app_state.visualize:
            from misc.time_plot import TimePlot
            self.plot = TimePlot()

    def plot_sequence(self, output_seq, data_tuple):
        """ Creates a default interactive visualization, with a slider enabling to move forth and back along the time axis (iteration in a given episode).
        The default visualizatoin contains input, output and target sequences.
        For more model/problem dependent visualization please overwrite this method in the derived model class.
        """
        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker
        from matplotlib import rc
        
        # Change fonts globally - for all figures at once.
        rc('font',**{'family':'Times New Roman'})
        
        # Change to np arrays and transpose, so x will be time axis.
        input_seq = data_tuple.inputs[0].cpu().detach().numpy()
        target_seq = data_tuple.targets[0].cpu().detach().numpy()
        output_seq = output_seq[0].cpu().detach().numpy()

        x = np.transpose(np.zeros(input_seq.shape))
        y = np.transpose(np.zeros(output_seq.shape))
        z = np.transpose(np.zeros(target_seq.shape))
        
        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info("Generating dynamic visualization of {} figures, please wait...".format(input_seq.shape[0]))
        # List of figures.
        figs = []
        for i, (input_word, output_word, target_word) in enumerate(zip(input_seq, output_seq, target_seq)):
            # Display information every 10% of figures.
            if (input_seq.shape[0] > 10) and (i % (input_seq.shape[0]//10) == 0):
                logger.info("Generating figure {}/{}".format(i, input_seq.shape[0]))
            fig = Figure()
            axes = fig.subplots(3, 1, sharex=True, sharey=False,
                                gridspec_kw={'width_ratios': [input_seq.shape[0]]})

            # Set ticks.
            axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            axes[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            axes[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            axes[2].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            # Set labels.
            axes[0].set_title('Inputs') 
            axes[0].set_ylabel('Control/Data bits')     
            axes[1].set_title('Targets')
            axes[1].set_ylabel('Data bits')
            axes[2].set_title('Predictions')
            axes[2].set_ylabel('Data bits')
            axes[2].set_xlabel('Item number')

            # Add words to adequate positions.
            x[:, i] = input_word
            y[:, i] = target_word
            z[:, i] = output_word
            # "Show" data on "axes".
            axes[0].imshow(x, interpolation='nearest', aspect='auto')
            axes[1].imshow(y, interpolation='nearest', aspect='auto')
            axes[2].imshow(z, interpolation='nearest', aspect='auto')
            # Append figure to a list.
            fig.set_tight_layout(True)
            figs.append(fig)

        # Set figure list to plot.
        self.plot.update(figs)
        return self.plot.is_closed
