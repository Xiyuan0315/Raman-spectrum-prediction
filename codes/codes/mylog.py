import matplotlib.pyplot as plt    
# from matplotlib.collections import EventCollection

"""
    Tools for logging
"""
import logging
import os

def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info('\n------ ******* ------ New Log ------ ******* ------')
    return logger


def log_text(file, message):
    """
        Docs:
            append new message into file
        Args:
            file: file to save message
            message: str
    """
    with open(file,'a') as f:
        f.write(message)
        f.write('\n')


def log_figure(file, data):
    """
        Docs: plot a figure with data and saved in file
        Args:
            file: pdf prefered
            data: dic with keys including 'x', 'y', 'color', 'title', 'legend'
    """
    x = data['x']
    y = data['y']
    color = data['color']
    title = data['title']
    label = data['label']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, color, label=label)
    ax.legend(loc='upper left')
    ax.set_title(title)
    fig.savefig(file)

# def log_figure_multilines(file, num=2, data1, data2):
#     """
#         Docs: plot a figure with data and saved in file
#         Args:
#             file: pdf prefered
#             data: dic with keys including 'x', 'y', 'color', 'title', 'legend'
#     """
#     n_lines = num
#     x1 = data1['x']
#     y1 = data1['y']
#     c1 = data1['color']
#     title1 = data1['title']
#     label1 = data1['legend']

#     x2 = data2['x']
#     y2 = data2['y']
#     c2 = data2['color']
#     title2 = data2['title']
#     label2 = data2['legend']

#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     ax.plot(x1, y1, c1, label=label1)
#     ax.plot(x2, y2, c2, label=label2)

#     ax.legend(loc='upper left')
#     ax.set_title(title1)
#     fig.savefig(file)
def log_figure_multilines(file, data):
    """
        Docs: plot a figure with data and saved in file
        Args:
            file: pdf prefered
            data: dic with keys including 'x', 'y', 'color', 'title', 'legend'
    """
    n_lines = data['n_lines']
    title = data['title']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(n_lines):
        x = data['x'+str(i+1)]
        y = data['y'+str(i+1)]
        color = data['color'+str(i+1)]
        label = data['label'+str(i+1)]
        ax.plot(x, y, color, label=label)        
    
    ax.legend(loc='upper left')
    ax.set_title(title)
    fig.savefig(file)
    


if __name__ == '__main__':
    # test log_figure
    import numpy as np

    # create random data
    xdata = np.random.random([2, 10])
    
    # split the data into two parts
    xdata1 = xdata[0, :]  
    # sort the data so it makes clean curves
    xdata1.sort()
    # create some y data points
    ydata1 = xdata1 ** 2

    # plot the data
    data = {'x':xdata1, 'y':ydata1, 'cmap':'b', 'title':'test figure', 'label':'test'}
    log_figure(file='./_fig.pdf', data=data)