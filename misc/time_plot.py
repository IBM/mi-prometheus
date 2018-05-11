import numpy as np
import sys
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

Qt = QtCore.Qt

class TimePlot(QtWidgets.QMainWindow):
    def __init__(self):
        self.qapp = QtWidgets.QApplication(sys.argv)
        super(TimePlot, self).__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QGridLayout(self._main)

        # Stacked plots: creates one widget per timestep. Shows only one at a time
        self.stacked_plots = QtWidgets.QStackedWidget()
        self.static_canvas = None
        self.figs = None
        self.layout.addWidget(self.stacked_plots, 0, 0)

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
        next_btn = QtWidgets.QPushButton("&Next episode")  # Shortcut is Alt+N
        next_btn.clicked.connect(self._next_clicked)
        quit_btn = QtWidgets.QPushButton("&Quit")  # Shortcut is Alt+Q
        quit_btn.clicked.connect(self.closeEvent)
        hbox_buttons.addWidget(next_btn)
        hbox_buttons.addWidget(quit_btn)
        self.layout.addLayout(hbox_buttons, 2, 0)

        self.is_closed = False
        self.is_playing = False

    def _next_clicked(self):
        self.qapp.quit()

    def closeEvent(self, _):
        self.is_closed = True
        self.qapp.quit()  # break event loop

    def update(self, figs: [Figure]):
        self.figs = [FigureCanvas(f) for f in figs]
        self.slider.setMaximum(len(figs) - 1)
        self.slider.setValue(len(figs) - 1)

        self.layout.removeWidget(self.stacked_plots)
        self.stacked_plots = QtWidgets.QStackedWidget()
        for f in self.figs:
            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(NavigationToolbar(f, self))
            layout.addWidget(f)
            w = QtWidgets.QWidget()
            w.setLayout(layout)
            self.stacked_plots.addWidget(w)
        self.stacked_plots.setCurrentIndex(len(figs) - 1)
        self.layout.addWidget(self.stacked_plots, 0, 0)
        self.show()
        self.qapp.exec_()  # Resume event loop

    def slider_valuechanged(self):
        val = self.slider.value()
        self.stacked_plots.setCurrentIndex(val)


if __name__ == '__main__':
    # Test code
    plot = TimePlot()

    while not plot.is_closed:
        x = np.zeros((15, 8))
        y = np.zeros((15, 8))
        z = np.zeros((15, 8))

        figs = []

        for i in range(15):
            fig = Figure()
            axes = fig.subplots(1, 3, sharey=True)

            x[i] = np.random.binomial(1, 0.5, 8)
            y[i] = np.random.binomial(1, 0.5, 8)
            z[i] = np.random.binomial(1, 0.5, 8)
            axes[0].imshow(x)
            axes[0].set_title("input")
            axes[1].imshow(y)
            axes[1].set_title("output")
            axes[2].imshow(z)
            axes[2].set_title("target")
            figs.append(fig)

        plot.update(figs)
