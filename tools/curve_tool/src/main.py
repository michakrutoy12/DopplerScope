import sys

from PySide6 import QtWidgets

from .qt_curve_tool import CurveTool


def main():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CurveTool()
    window.show()
    return app.exec()


if __name__ == "__main__":
    main()
