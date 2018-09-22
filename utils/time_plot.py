import sys
from datetime import datetime
import numpy as np
import logging

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

#Qt = QtCore.Qt


class TimePlot(QtWidgets.QMainWindow):
    def __init__(self):
        self.qapp = QtWidgets.QApplication(sys.argv)
        super(TimePlot, self).__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QGridLayout(self._main)

        # Empty objects that will be used during visualization.
        self.fig = None
        self.frames = None

        # Slider stuff
        hbox_timeline = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.slider_valuechanged)
        hbox_timeline.addWidget(self.slider)
        self.layout.addLayout(hbox_timeline, 1, 0)

        # Bottom buttons
        hbox_buttons = QtWidgets.QHBoxLayout()
        hbox_buttons.addStretch(1)
        # Button for saving movies.
        save_movie_btn = QtWidgets.QPushButton(
            "&Save as movie")  # Shortcut is Alt+S
        save_movie_btn.clicked.connect(self._save_movie_clicked)
        # Button for next episode.
        next_btn = QtWidgets.QPushButton("&Next episode")  # Shortcut is Alt+N
        next_btn.clicked.connect(self._next_clicked)
        # Quit button.
        quit_btn = QtWidgets.QPushButton("&Quit")  # Shortcut is Alt+Q
        quit_btn.clicked.connect(self.closeEvent)
        # Add buttons to widget.
        hbox_buttons.addWidget(save_movie_btn)
        hbox_buttons.addWidget(next_btn)
        hbox_buttons.addWidget(quit_btn)
        self.layout.addLayout(hbox_buttons, 2, 0)

        self.is_closed = False
        self.is_playing = False

    def _save_movie_clicked(self):
        logger = logging.getLogger('ModelBase')
        logger.info("Saving movie - please wait...")
        # Generate filename.
        time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
        name_str = 'experiment_run_' + time_str + '.mp4'

        # Save the animation.
        ani = animation.ArtistAnimation(
            self.fig, self.frames, blit=False, interval=1.0)
        Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=1, extra_args=['-r', '25', '-probesize', '10M'])
        # ani.save(name_str, writer=writer, dpi=200)
        writer = Writer(fps=1, bitrate=5000)
        ani.save(name_str, writer=writer)
        logger.info("Saved movie to file '{}'".format(name_str))

    def _next_clicked(self):
        self.qapp.quit()

    def closeEvent(self, _):
        # Set flag to True, so we could break the external loop.
        self.is_closed = True
        self.qapp.quit()

    def update(self, fig, frames):
        # "Save" figure and frame objects.
        self.fig = fig
        self.frames = frames

        # Create the widget.
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        # Cast to figure canvas.
        fc = FigureCanvas(fig)
        # Add "navigation" (save) on top.
        layout.addWidget(NavigationToolbar(fc, self))
        layout.addWidget(fc)
        w.setLayout(layout)
        # Add widget at position (0,0)!
        self.layout.addWidget(w, 0, 0)
        # Reset current frame.
        self.current_frame_number = 0

        # Set slider properties...
        self.slider.setMaximum(len(frames) - 1)
        # ... And set slider to end, which will cause showing last frame.
        self.slider.setValue(len(frames) - 1)
        # But call it anyway - for the case when user pressed "next" button.
        self.show_frame(len(frames) - 1)

        # Show.
        self.show()
        self.qapp.exec_()  # Resume event loop

    def slider_valuechanged(self):
        """
        Event handler attached to the slider.
        """
        val = self.slider.value()
        self.show_frame(val)

    def show_frame(self, frame_number):
        """
        Shows given frame.
        """
        # Make all the artists from the current frame visible
        for frame in self.frames:
            for artist in frame:  # self.frames[self.current_frame_number]:
                artist.set_visible(False)

        # Make all the artists from the current frame visible
        for artist in self.frames[frame_number]:
            artist.set_visible(True)

        # Redraw the figure canvas.
        self.fig.canvas.draw_idle()

        # Remember current frame.
        self.current_frame_number = frame_number


if __name__ == '__main__':
    # Set logging level.
    logging.basicConfig(level=logging.DEBUG)

    # Test code
    plot = TimePlot()

    # Create a single "figure canvas".
    fig = plt.figure()
    axes = fig.subplots(1, 3, sharey=True)
    axes[0].set_title("input")
    axes[1].set_title("output")
    axes[2].set_title("target")

    def f(x, y):
        return np.sin(x) + np.cos(y)

    while not plot.is_closed:
        x = np.linspace(0, 2 * np.pi, 120)
        y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
        # frames is a list of lists, each row is a list of artists to draw in a
        # given frame.
        frames = []
        for i in range(60):
            x += np.pi / 15.
            y += np.pi / 20.
            # Get axes
            artists = [None] * len(fig.axes)
            artists[0] = axes[0].imshow(f(x, y))
            artists[1] = axes[1].imshow(f(x, y))
            artists[2] = axes[2].imshow(f(x, y))
            # Make them invisible.
            for artist in artists:
                artist.set_visible(False)
            # Add "frame".
            frames.append(artists)
        # Plot.
        plot.update(fig, frames)
