import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import sleep
import numpy as np

def plot_memory_attention(memory, wt):
    plt.clf()
    fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylabel("Word size", fontname='Times New Roman', fontsize=15)
    ax2.set_xlabel("Memory addresses", fontname='Times New Roman', fontsize=15)

    ax2.imshow(memory[0,:,:].detach().numpy(), interpolation='nearest')

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_ylabel("Attention", fontname='Times New Roman', fontsize=15)
    ax1.set_title('Scratch Pad task, seq_lengths: ', fontname='Times New Roman', fontsize=15)

    ax1.plot(np.arange(wt.size()[-1]), wt[0, 1, :].detach().numpy(), 'go')

    plt.pause(0.01)