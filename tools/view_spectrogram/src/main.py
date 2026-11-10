import sys

from PySide6 import QtWidgets

from .viewer import SpectrogramViewer


def main():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = SpectrogramViewer()
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    main()
