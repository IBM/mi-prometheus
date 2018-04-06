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

# mem_t = np.ones((1, 8, 60)) * 0.01
#
# i=0
# while True:
#     for i in range(i, i+3):
#         seq = np.random.binomial(1, 0.5, (1, 1, 8))
#         mem_t[:,:,i] = seq
#         plot_memory(mem_t)
#
#     for j in range(5):
#         seq = np.random.binomial(1, 0.5, (1, 1, 8))
#         mem_t[:,:,j+45] = seq
#         plot_memory(mem_t)
#         if i == 40:
#             break


#ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
#patch = ax.arrow(0, 0, 0.5, 0.5, head_width=0.1, head_length=0.1, fc='k', ec='k')
