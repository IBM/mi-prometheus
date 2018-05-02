import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import sleep
import numpy as np

def plot_memory_attention(prediction, memory, wt, label):
    plt.clf()
    fig = plt.figure(1)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(212)
    ax3 = fig.add_subplot(222)

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_title("Attention", fontname='Times New Roman', fontsize=15)
    ax1.plot(np.arange(wt.size()[-1]), wt[0, 0, :].detach().numpy(), 'go')

    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylabel("Word size", fontname='Times New Roman', fontsize=15)
    ax2.set_xlabel("Memory addresses", fontname='Times New Roman', fontsize=15)
    ax2.set_title("Task: xxx", fontname='Times New Roman', fontsize=15)

    ax2.imshow(memory[0,:,:].detach().numpy(), interpolation='nearest')

    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.set_title("Prediction", fontname='Times New Roman', fontsize=15)
    ax3.imshow(np.transpose((prediction[0, ...]).detach().numpy(), [1, 0]))

    plt.pause(0.1)
    input('press key to continue ...')

def plot_memory(memory):
    plt.clf()
    fig = plt.figure(1)

    ax2 = fig.add_subplot(111)

    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylabel("Word size", fontname='Times New Roman', fontsize=15)
    ax2.set_xlabel("Memory addresses", fontname='Times New Roman', fontsize=15)

    ax2.imshow(memory[0,:,:], interpolation='nearest', cmap='Greys')
    plt.pause(0.1)
    #input("pause")