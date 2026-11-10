import csv
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


class CurveViewBox(pg.ViewBox):
    pointAdded = QtCore.Signal(float, float)
    pointMoved = QtCore.Signal(int, float, float)
    pointSelected = QtCore.Signal(int)

    def __init__(self):
        super().__init__(enableMenu=False)
        self.add_mode = True
        self.points = []
        self._drag_index = None
        self._data_bounds = None

    def set_curve_points(self, points):
        self.points = points

    def set_data_bounds(self, time_bounds, frequency_bounds):
        self._data_bounds = (time_bounds, frequency_bounds)

    def _nearest_point(self, scene_position):
        nearest_index = None
        nearest_distance = 12.0
        for index, (time_value, frequency_value) in enumerate(self.points):
            scene_point = self.mapViewToScene(
                pg.Point(time_value, frequency_value)
            )
            distance = math.hypot(
                scene_point.x() - scene_position.x(),
                scene_point.y() - scene_position.y(),
            )
            if distance <= nearest_distance:
                nearest_index = index
                nearest_distance = distance
        return nearest_index

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.add_mode:
                point = self.mapSceneToView(event.scenePos())
                self.pointAdded.emit(point.x(), point.y())
                event.accept()
                return

            index = self._nearest_point(event.scenePos())
            if index is not None:
                self.pointSelected.emit(index)
                event.accept()
                return
        super().mouseClickEvent(event)

    def mouseDragEvent(self, event, axis=None):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if event.isStart():
                self._drag_index = self._nearest_point(event.scenePos())
                if self._drag_index is not None:
                    self.pointSelected.emit(self._drag_index)

            if self._drag_index is not None:
                point = self.mapSceneToView(event.scenePos())
                if self._data_bounds is not None:
                    (min_time, max_time), (min_frequency, max_frequency) = (
                        self._data_bounds
                    )
                    point.setX(min(max(point.x(), min_time), max_time))
                    point.setY(
                        min(max(point.y(), min_frequency), max_frequency)
                    )
                self.pointMoved.emit(
                    self._drag_index, point.x(), point.y()
                )
                if event.isFinish():
                    self._drag_index = None
                event.accept()
                return
        super().mouseDragEvent(event, axis=axis)


class CurveTool(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram Curve Tool")
        self.resize(1250, 800)
        self.setMinimumSize(850, 550)

        self.file_path = None
        self.spectrogram = None
        self.time_edges = None
        self.frequency_edges = None
        self.points = []
        self.selected_index = None

        self._build_ui()
        self._set_loaded_controls_enabled(False)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setSpacing(6)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        self.open_button = QtWidgets.QPushButton("Open snapshot…")
        controls.addWidget(self.open_button)

        controls.addWidget(QtWidgets.QLabel("Dynamic range:"))
        self.dynamic_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.dynamic_slider.setRange(20, 120)
        self.dynamic_slider.setValue(80)
        self.dynamic_slider.setMaximumWidth(240)
        controls.addWidget(self.dynamic_slider)
        self.dynamic_label = QtWidgets.QLabel("80 dB")
        controls.addWidget(self.dynamic_label)
        controls.addStretch(1)

        navigation = QtWidgets.QHBoxLayout()
        layout.addLayout(navigation)
        self.add_mode_button = QtWidgets.QPushButton("Add points")
        self.add_mode_button.setCheckable(True)
        self.add_mode_button.setChecked(True)
        navigation.addWidget(self.add_mode_button)
        self.delete_button = QtWidgets.QPushButton("Delete selected")
        navigation.addWidget(self.delete_button)
        self.clear_button = QtWidgets.QPushButton("Clear curve")
        navigation.addWidget(self.clear_button)
        navigation.addSpacing(12)
        self.zoom_in_button = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_button = QtWidgets.QPushButton("Zoom −")
        self.fit_button = QtWidgets.QPushButton("Fit")
        navigation.addWidget(self.zoom_in_button)
        navigation.addWidget(self.zoom_out_button)
        navigation.addWidget(self.fit_button)
        navigation.addSpacing(12)
        self.export_button = QtWidgets.QPushButton("Export curve…")
        navigation.addWidget(self.export_button)
        navigation.addWidget(
            QtWidgets.QLabel(
                "Wheel: zoom · drag: pan · right-drag: box zoom · "
                "click: add point · drag point: move"
            )
        )
        navigation.addStretch(1)

        self.view_box = CurveViewBox()
        self.plot = pg.PlotWidget(viewBox=self.view_box)
        self.plot.setBackground("#17191f")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setTitle("Open a spectrogram snapshot pickle")
        self.view_box.setMouseEnabled(x=True, y=True)

        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setAutoDownsample(True)
        self.plot.addItem(self.image)
        color_map = pg.colormap.get("magma")
        self.image.setLookupTable(color_map.getLookupTable(0.0, 1.0, 256))
        self.color_bar = pg.ColorBarItem(
            values=(-80, 0),
            colorMap=color_map,
            label="Amplitude (dBFS)",
            interactive=False,
        )
        self.color_bar.setImageItem(self.image)
        self.color_bar_widget = pg.GraphicsLayoutWidget()
        self.color_bar_widget.setMinimumWidth(75)
        self.color_bar_widget.setMaximumWidth(105)
        self.color_bar_widget.ci.addItem(self.color_bar)

        self.curve_item = self.plot.plot([], [], pen=pg.mkPen("#4de1c1", width=2))
        self.points_item = pg.ScatterPlotItem(
            pxMode=True,
            size=10,
            pen=pg.mkPen("#17191f", width=1),
            brush=pg.mkBrush("#4de1c1"),
        )
        self.selected_point_item = pg.ScatterPlotItem(
            pxMode=True,
            size=14,
            pen=pg.mkPen("#ffffff", width=2),
            brush=pg.mkBrush("#ffb454"),
        )
        self.plot.addItem(self.points_item)
        self.plot.addItem(self.selected_point_item)

        plot_row = QtWidgets.QHBoxLayout()
        plot_row.addWidget(self.plot, stretch=1)
        plot_row.addWidget(self.color_bar_widget)
        layout.addLayout(plot_row, stretch=1)

        self.status = QtWidgets.QLabel("Ready")
        self.status.setStyleSheet("color: #aab2c0; padding: 2px 0;")
        layout.addWidget(self.status)

        self.open_button.clicked.connect(self.open_file)
        self.dynamic_slider.valueChanged.connect(self.on_dynamic_change)
        self.add_mode_button.toggled.connect(self._set_add_mode)
        self.delete_button.clicked.connect(self.delete_selected_point)
        self.clear_button.clicked.connect(self.clear_curve)
        self.zoom_in_button.clicked.connect(lambda: self._zoom(0.7))
        self.zoom_out_button.clicked.connect(lambda: self._zoom(1.4))
        self.fit_button.clicked.connect(self.fit_data)
        self.export_button.clicked.connect(self.export_curve)
        self.view_box.pointAdded.connect(self.add_point)
        self.view_box.pointMoved.connect(self.move_point)
        self.view_box.pointSelected.connect(self.select_point)

    def _set_loaded_controls_enabled(self, enabled):
        for widget in (
            self.dynamic_slider,
            self.add_mode_button,
            self.delete_button,
            self.clear_button,
            self.zoom_in_button,
            self.zoom_out_button,
            self.fit_button,
            self.export_button,
        ):
            widget.setEnabled(enabled)
        self.delete_button.setEnabled(
            enabled and self.selected_index is not None
        )
        self.clear_button.setEnabled(enabled and bool(self.points))
        self.export_button.setEnabled(enabled and bool(self.points))

    @staticmethod
    def _load_snapshot(path):
        with open(path, "rb") as input_file:
            snapshot = pickle.load(input_file)
        if not isinstance(snapshot, dict):
            raise ValueError("The pickle must contain a spectrogram dictionary.")
        try:
            matrix = np.asarray(snapshot["spectrogram_db"], dtype=np.float32)
            time_edges = np.asarray(snapshot["time_edges"], dtype=np.float64)
            frequency_edges = np.asarray(
                snapshot["frequency_edges"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "The snapshot must contain spectrogram_db, time_edges, "
                "and frequency_edges arrays."
            ) from exc

        if matrix.ndim != 2:
            raise ValueError("spectrogram_db must be a two-dimensional array.")
        if time_edges.ndim != 1 or frequency_edges.ndim != 1:
            raise ValueError("Time and frequency edges must be one-dimensional.")
        if matrix.shape != (
            frequency_edges.size - 1,
            time_edges.size - 1,
        ):
            raise ValueError(
                "The spectrogram dimensions do not match its time and "
                "frequency edges."
            )
        if time_edges.size < 2 or frequency_edges.size < 2:
            raise ValueError("The snapshot must contain at least one data cell.")
        if (
            not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(time_edges))
            or not np.all(np.isfinite(frequency_edges))
        ):
            raise ValueError("The snapshot contains non-finite values.")
        if np.any(np.diff(time_edges) <= 0) or np.any(
            np.diff(frequency_edges) <= 0
        ):
            raise ValueError("Time and frequency edges must be strictly increasing.")
        return matrix, time_edges, frequency_edges

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open spectrogram snapshot (trusted files only)",
            "",
            "Pickle files (*.pkl *.pickle);;All files (*)",
        )
        if not path:
            return
        try:
            matrix, time_edges, frequency_edges = self._load_snapshot(path)
        except (
            OSError,
            EOFError,
            pickle.PickleError,
            ImportError,
            AttributeError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
        ) as exc:
            QtWidgets.QMessageBox.critical(
                self, "Could not open snapshot", str(exc)
            )
            return

        self.file_path = path
        self.spectrogram = matrix
        self.time_edges = time_edges
        self.frequency_edges = frequency_edges
        self.points = []
        self.selected_index = None
        self.view_box.set_curve_points(self.points)
        self.view_box.set_data_bounds(
            (float(time_edges[0]), float(time_edges[-1])),
            (float(frequency_edges[0]), float(frequency_edges[-1])),
        )
        self.image.setImage(matrix, autoLevels=False)
        self.image.setRect(
            QtCore.QRectF(
                float(time_edges[0]),
                float(frequency_edges[0]),
                float(time_edges[-1] - time_edges[0]),
                float(frequency_edges[-1] - frequency_edges[0]),
            )
        )
        self.on_dynamic_change(self.dynamic_slider.value())
        self.plot.setTitle(os.path.basename(path))
        self.dynamic_slider.setValue(80)
        self._set_loaded_controls_enabled(True)
        self._set_add_mode(self.add_mode_button.isChecked())
        self._refresh_curve()
        self.fit_data()
        self.status.setText(
            f"{os.path.basename(path)} | {matrix.shape[1]} time columns × "
            f"{matrix.shape[0]} frequency rows"
        )

    def on_dynamic_change(self, value):
        self.dynamic_label.setText(f"{value} dB")
        if self.spectrogram is None:
            return
        maximum = float(np.max(self.spectrogram))
        levels = (maximum - float(value), maximum)
        self.image.setLevels(levels)
        self.color_bar.setLevels(values=levels)

    def _set_add_mode(self, enabled):
        self.view_box.add_mode = enabled
        self.plot.setCursor(
            QtCore.Qt.CursorShape.CrossCursor
            if enabled
            else QtCore.Qt.CursorShape.ArrowCursor
        )

    def add_point(self, time_value, frequency_value):
        min_time, max_time = float(self.time_edges[0]), float(self.time_edges[-1])
        min_frequency = float(self.frequency_edges[0])
        max_frequency = float(self.frequency_edges[-1])
        self.points.append(
            (
                min(max(float(time_value), min_time), max_time),
                min(max(float(frequency_value), min_frequency), max_frequency),
            )
        )
        self.selected_index = len(self.points) - 1
        self._refresh_curve()

    def move_point(self, index, time_value, frequency_value):
        if not 0 <= index < len(self.points):
            return
        self.points[index] = (float(time_value), float(frequency_value))
        self._refresh_curve()

    def select_point(self, index):
        if 0 <= index < len(self.points):
            self.selected_index = index
            self._refresh_curve()

    def delete_selected_point(self):
        if self.selected_index is None:
            return
        del self.points[self.selected_index]
        self.selected_index = (
            min(self.selected_index, len(self.points) - 1)
            if self.points
            else None
        )
        self._refresh_curve()

    def clear_curve(self):
        self.points.clear()
        self.selected_index = None
        self._refresh_curve()

    def _refresh_curve(self):
        self.view_box.set_curve_points(self.points)
        if self.points:
            points = np.asarray(self.points, dtype=np.float64)
            order = np.argsort(points[:, 0], kind="stable")
            self.curve_item.setData(points[order, 0], points[order, 1])
            self.points_item.setData(points[:, 0], points[:, 1])
        else:
            self.curve_item.setData([], [])
            self.points_item.setData([], [])

        if self.selected_index is None:
            self.selected_point_item.setData([], [])
        else:
            selected = self.points[self.selected_index]
            self.selected_point_item.setData([selected[0]], [selected[1]])
        self.delete_button.setEnabled(self.selected_index is not None)
        self.clear_button.setEnabled(bool(self.points))
        self.export_button.setEnabled(
            self.spectrogram is not None and len(self.points) >= 2
        )
        self.status.setText(f"{len(self.points)} curve points")

    def _zoom(self, factor):
        self.view_box.scaleBy((factor, factor))

    def fit_data(self):
        if self.time_edges is None:
            return
        self.view_box.setRange(
            xRange=(float(self.time_edges[0]), float(self.time_edges[-1])),
            yRange=(
                float(self.frequency_edges[0]),
                float(self.frequency_edges[-1]),
            ),
            padding=0.02,
        )

    def export_curve(self):
        if len(self.points) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "Not enough points",
                "Add at least two points before exporting a curve.",
            )
            return
        default_path = Path(__file__).resolve().parents[1] / "observed_curve.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export observed curve",
            str(default_path),
            "CSV files (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        points = np.asarray(self.points, dtype=np.float64)
        order = np.argsort(points[:, 0], kind="stable")
        obs_time = points[order, 0]
        obs_freq = points[order, 1]
        try:
            with open(path, "w", newline="", encoding="utf-8") as output_file:
                writer = csv.writer(output_file)
                writer.writerow(("time_s", "frequency_hz"))
                writer.writerows(zip(obs_time, obs_freq))
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not export curve data:\n{exc}"
            )
            return
        self.status.setText(
            f"Exported {len(obs_time)} points to {os.path.basename(path)} "
            "(time_s, frequency_hz)"
        )
