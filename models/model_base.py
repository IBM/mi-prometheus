import numpy as np
import logging
import torch

# Add path to main project directory - so we can test the base plot, saving images, movies etc.
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..')) 
from misc.app_state import AppState
from problems.algorithmic_sequential_problem import DataTuple

class ModelBase(object):
    """ Class representing base class of all models.
    Provides basic plotting functionality.
    """
    def __init__(self):
        """ Initializes application state and sets plot if visualization flag is turned on."""
        super(ModelBase, self).__init__()
        # WARNING: at that moment AppState must be initialized and flag must be set. Otherwise the object plot won't be created.
        # SOLUTION: if application is supposed to show dynamic plot, set flag to True before constructing the model! (and set to False right after if required)
        self.app_state = AppState()

        if self.app_state.visualize:
            from misc.time_plot import TimePlot
            self.plot = TimePlot()

    def plot_sequence(self, data_tuple, predictions_seq):
        """ Creates a default interactive visualization, with a slider enabling to move forth and back along the time axis (iteration in a given episode).
        The default visualizatoin contains input, output and target sequences.
        For more model/problem dependent visualization please overwrite this method in the derived model class.
        
        :param data_tuple: Data tuple containing input [BATCH_SIZE x SEQUENCE_LENGTH x INPUT_DATA_SIZE] and target sequences  [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_DATA_SIZE]
        :param predictions_seq: Prediction sequence [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_DATA_SIZE]
        """
        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker

        # Change fonts globally - for all figures/subsplots at once.
        #from matplotlib import rc
        #rc('font', **{'family': 'Times New Roman'})
        import matplotlib.pylab as pylab
        params = {
        #'legend.fontsize': '28',
         'axes.titlesize':'large',
         'axes.labelsize': 'large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
        pylab.rcParams.update(params)

        # Create a single "figure layout" for all displayed frames.
        fig = Figure()
        axes = fig.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'width_ratios': [predictions_seq.shape[0]]})

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
 
        fig.set_tight_layout(True)
        
        # Change to np arrays and transpose, so x will be time axis.
        inputs_seq = data_tuple.inputs[0].cpu().detach().numpy()
        targets_seq = data_tuple.targets[0].cpu().detach().numpy()
        predictions_seq = predictions_seq[0].cpu().detach().numpy()
        print(inputs_seq.shape)

        # Create empty matrices.
        x = np.transpose(np.zeros(inputs_seq.shape))
        y = np.transpose(np.zeros(predictions_seq.shape))
        z = np.transpose(np.zeros(targets_seq.shape))
        
        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info("Generating dynamic visualization of {} figures, please wait...".format(inputs_seq.shape[0]))
        
        # Create frames - a list of lists, where each row is a list of artists used to draw a given frame.
        frames = []

        for i, (input_word, prediction_word, target_word) in enumerate(zip(inputs_seq, predictions_seq, targets_seq)):
            # Display information every 10% of figures.
            if (inputs_seq.shape[0] > 10) and (i % (inputs_seq.shape[0]//10) == 0):
                logger.info("Generating figure {}/{}".format(i, inputs_seq.shape[0]))

            # Add words to adequate positions.
            x[:, i] = input_word
            y[:, i] = target_word
            z[:, i] = prediction_word
            
            # Create "Artists" drawing data on "ImageAxes".
            artists = [None] * len( fig.axes)
            
            # Tell artists what to do;)
            artists[0] = axes[0].imshow(x, interpolation='nearest', aspect='auto')
            artists[1] = axes[1].imshow(y, interpolation='nearest', aspect='auto')
            artists[2] = axes[2].imshow(z, interpolation='nearest', aspect='auto')
                
            # Add "frame".
            frames.append(artists)

        # Plot figure and list of frames.
        self.plot.update(fig,  frames)
        
        # Return True if user closed the window.
        return self.plot.is_closed

if __name__ == '__main__':
    # Set logging level.
    logging.basicConfig(level=logging.DEBUG)
    
    # Set visualization.
    AppState().visualize = True
    
    # Test code
    test = ModelBase()
    
    while not test.plot.is_closed:
        # Generate new sequence.
        x = np.random.binomial(1, 0.5, (1,  8,  15))
        y = np.random.binomial(1, 0.5, (1,  8,  15))
        z = np.random.binomial(1, 0.5, (1,  8, 15))
        print(x.shape)
        # Transform to PyTorch.
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y=  torch.from_numpy(y).type(torch.FloatTensor)
        z=  torch.from_numpy(z).type(torch.FloatTensor)
        dt = DataTuple(x, y)
        # Plot it
        test.plot_sequence(dt, z)
