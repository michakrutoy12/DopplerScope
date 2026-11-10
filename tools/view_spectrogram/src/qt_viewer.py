import math
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PySide6 import QtCore, QtWidgets
from scipy.signal import sosfilt

try:
    import sounddevice as sd
except ImportError:
    sd = None

from .playback import build_playback_filter, format_time
from .spectrogram import calculate_visible_spectrogram


class SpectrogramViewBox(pg.ViewBox):
    rangeSelected = QtCore.Signal(float, float, bool)

    def __init__(self):
        super().__init__(enableMenu=False)
        self.selection_enabled = False
        self._selection_start = None

    def mouseDragEvent(self, event, axis=None):
        if (
            self.selection_enabled
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            x = self.mapSceneToView(event.scenePos()).x()
            if event.isStart():
                self._selection_start = x
            if self._selection_start is not None:
                self.rangeSelected.emit(
                    self._selection_start, x, event.isFinish()
                )
            if event.isFinish():
                self._selection_start = None
            event.accept()
            return
        super().mouseDragEvent(event, axis=axis)


class PlaybackWorker(QtCore.QThread):
    def __init__(
        self,
        file_path,
        sample_rate,
        start_sample,
        channel_mode,
        filter_sos,
        parent=None,
    ):
        super().__init__(parent)
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.position_samples = start_sample
        self.channel_mode = channel_mode
        self.filter_sos = filter_sos
        self._stop_event = threading.Event()
        self.error = None

    def request_stop(self):
        self._stop_event.set()

    def run(self):
        filter_state = (
            None
            if self.filter_sos is None
            else np.zeros((self.filter_sos.shape[0], 2, 1), dtype=np.float32)
        )
        try:
            with sf.SoundFile(self.file_path, mode="r") as audio_file:
                audio_file.seek(self.position_samples)
                with sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=2,
                    dtype="float32",
                    blocksize=0,
                ) as stream:
                    while not self._stop_event.is_set():
                        data = audio_file.read(
                            1024, dtype="float32", always_2d=True
                        )
                        if not len(data):
                            break
                        if self.channel_mode == "Left":
                            mono = data[:, 0]
                        elif self.channel_mode == "Right":
                            mono = data[:, min(1, data.shape[1] - 1)]
                        elif data.shape[1] == 1:
                            mono = data[:, 0]
                        else:
                            mono = np.mean(data, axis=1, dtype=np.float32)

                        mono = np.asarray(mono, dtype=np.float32).reshape(-1, 1)
                        if self.filter_sos is not None:
                            mono, filter_state = sosfilt(
                                self.filter_sos,
                                mono,
                                axis=0,
                                zi=filter_state,
                            )
                        stream.write(np.repeat(mono, 2, axis=1))
                        if not self._stop_event.is_set():
                            self.position_samples += len(data)
        except Exception as exc:
            if not self._stop_event.is_set():
                self.error = str(exc)


class SpectrogramViewer(QtWidgets.QMainWindow):
    FFT_CHOICES = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    HOP_CHOICES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Spectrogram Viewer")
        self.resize(1250, 820)
        self.setMinimumSize(850, 550)

        self.audio_file = None
        self.file_path = None
        self.sample_rate = 0
        self.total_samples = 0
        self.channel_count = 0
        self.duration = 0.0
        self._playback_position_samples = 0
        self._playback_worker = None
        self._is_playing = False
        self._resume_after_worker = False
        self._pending_seek_sample = None
        self._pending_open_path = None
        self._seeking = False
        self._syncing_controls = False
        self._selection_region = None
        self._render_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="spectrogram"
        )
        self._render_future = None
        self._render_cancel = None
        self._render_generation = 0
        self._displayed_spectrogram = None
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(160)
        self._render_timer.timeout.connect(self._refresh_visible_range)
        self._playback_timer = QtCore.QTimer(self)
        self._playback_timer.setInterval(40)
        self._playback_timer.timeout.connect(self._update_playback_position)

        self._build_ui()
        self._set_controls_enabled(False)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setSpacing(6)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        self.open_button = QtWidgets.QPushButton("Open audio…")
        controls.addWidget(self.open_button)

        controls.addWidget(QtWidgets.QLabel("FFT size"))
        self.fft_box = QtWidgets.QComboBox()
        self.fft_box.addItems([str(value) for value in self.FFT_CHOICES])
        self.fft_box.setCurrentText("4096")
        controls.addWidget(self.fft_box)

        controls.addWidget(QtWidgets.QLabel("Hop length"))
        self.hop_box = QtWidgets.QComboBox()
        self.hop_box.addItems([str(value) for value in self.HOP_CHOICES])
        self.hop_box.setCurrentText("512")
        controls.addWidget(self.hop_box)

        controls.addWidget(QtWidgets.QLabel("Channel"))
        self.channel_box = QtWidgets.QComboBox()
        self.channel_box.addItems(("Mono", "Left", "Right"))
        controls.addWidget(self.channel_box)

        controls.addWidget(QtWidgets.QLabel("Minimum frequency (Hz)"))
        self.min_freq_edit = QtWidgets.QLineEdit("0")
        self.min_freq_edit.setMaximumWidth(80)
        controls.addWidget(self.min_freq_edit)
        controls.addWidget(QtWidgets.QLabel("Maximum frequency (Hz)"))
        self.max_freq_edit = QtWidgets.QLineEdit("10000")
        self.max_freq_edit.setMaximumWidth(80)
        controls.addWidget(self.max_freq_edit)
        self.apply_button = QtWidgets.QPushButton("Apply")
        controls.addWidget(self.apply_button)
        controls.addStretch(1)

        settings = QtWidgets.QHBoxLayout()
        layout.addLayout(settings)
        settings.addWidget(QtWidgets.QLabel("Dynamic range:"))
        self.dynamic_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.dynamic_slider.setRange(20, 120)
        self.dynamic_slider.setValue(80)
        self.dynamic_slider.setMaximumWidth(240)
        settings.addWidget(self.dynamic_slider)
        self.dynamic_label = QtWidgets.QLabel("80 dB")
        settings.addWidget(self.dynamic_label)
        settings.addSpacing(18)
        settings.addWidget(QtWidgets.QLabel("Time range (seconds):"))
        settings.addWidget(QtWidgets.QLabel("From"))
        self.range_start_edit = QtWidgets.QLineEdit("0")
        self.range_start_edit.setMaximumWidth(110)
        settings.addWidget(self.range_start_edit)
        settings.addWidget(QtWidgets.QLabel("To"))
        self.range_end_edit = QtWidgets.QLineEdit("0")
        self.range_end_edit.setMaximumWidth(110)
        settings.addWidget(self.range_end_edit)
        self.show_range_button = QtWidgets.QPushButton("Show segment")
        settings.addWidget(self.show_range_button)
        self.full_range_button = QtWidgets.QPushButton("Full recording")
        settings.addWidget(self.full_range_button)
        settings.addStretch(1)

        navigation = QtWidgets.QHBoxLayout()
        layout.addLayout(navigation)
        self.select_range_button = QtWidgets.QPushButton("Select range")
        self.select_range_button.setCheckable(True)
        navigation.addWidget(self.select_range_button)
        self.export_button = QtWidgets.QPushButton("Export visible data…")
        self.export_button.setEnabled(False)
        navigation.addWidget(self.export_button)
        navigation.addWidget(
            QtWidgets.QLabel(
                "Wheel: zoom · drag: pan · right-drag: box zoom · "
                "Select range: drag to zoom to a segment · drag red line to seek"
            )
        )
        navigation.addStretch(1)

        self.view_box = SpectrogramViewBox()
        self.plot = pg.PlotWidget(viewBox=self.view_box)
        self.plot.setBackground("#17191f")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setTitle("Open a WAV, FLAC, OGG, AIFF or AU audio file")
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
        playhead_pen = pg.mkPen("#4de1c1", width=2)
        self.playhead = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=playhead_pen,
            hoverPen=pg.mkPen("#a0ffe9", width=3),
        )
        self.playhead.setVisible(False)
        self.plot.addItem(self.playhead)

        plot_row = QtWidgets.QHBoxLayout()
        plot_row.addWidget(self.plot, stretch=1)
        plot_row.addWidget(self.color_bar_widget)
        layout.addLayout(plot_row, stretch=1)

        seek_row = QtWidgets.QHBoxLayout()
        layout.addLayout(seek_row)
        self.time_label = QtWidgets.QLabel("00:00.0 / 00:00.0")
        self.time_label.setMinimumWidth(120)
        seek_row.addWidget(self.time_label)
        self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setTracking(True)
        seek_row.addWidget(self.seek_slider, stretch=1)

        player = QtWidgets.QHBoxLayout()
        layout.addLayout(player)
        self.play_button = QtWidgets.QPushButton("Play")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.stop_button = QtWidgets.QPushButton("Stop")
        player.addWidget(self.play_button)
        player.addWidget(self.pause_button)
        player.addWidget(self.stop_button)
        player.addSpacing(12)
        player.addWidget(QtWidgets.QLabel("Playback filter (Hz):"))
        player.addWidget(QtWidgets.QLabel("Low"))
        self.low_cut_edit = QtWidgets.QLineEdit("0")
        self.low_cut_edit.setMaximumWidth(80)
        player.addWidget(self.low_cut_edit)
        player.addWidget(QtWidgets.QLabel("High"))
        self.high_cut_edit = QtWidgets.QLineEdit("10000")
        self.high_cut_edit.setMaximumWidth(80)
        player.addWidget(self.high_cut_edit)
        self.filter_button = QtWidgets.QPushButton("Apply filter")
        player.addWidget(self.filter_button)
        player.addStretch(1)

        self.status = QtWidgets.QLabel("Ready")
        self.status.setStyleSheet("color: #aab2c0; padding: 2px 0;")
        layout.addWidget(self.status)

        self.open_button.clicked.connect(self.open_file)
        self.fft_box.currentIndexChanged.connect(self.apply_settings)
        self.hop_box.currentIndexChanged.connect(self.apply_settings)
        self.channel_box.currentIndexChanged.connect(self.apply_settings)
        self.min_freq_edit.returnPressed.connect(self.apply_settings)
        self.max_freq_edit.returnPressed.connect(self.apply_settings)
        self.apply_button.clicked.connect(self.apply_settings)
        self.export_button.clicked.connect(self.export_visible_spectrogram)
        self.dynamic_slider.valueChanged.connect(self.on_dynamic_change)
        self.show_range_button.clicked.connect(self.show_time_range)
        self.full_range_button.clicked.connect(self.show_full_recording)
        self.select_range_button.toggled.connect(self._set_selection_mode)
        self.view_box.rangeSelected.connect(self._on_range_selected)
        self.view_box.sigXRangeChanged.connect(self._on_x_range_changed)
        self.playhead.sigPositionChanged.connect(self._on_playhead_changed)
        self.playhead.sigPositionChangeFinished.connect(self._finish_seek)
        self.seek_slider.sliderPressed.connect(self._begin_seek)
        self.seek_slider.sliderMoved.connect(self._preview_seek)
        self.seek_slider.sliderReleased.connect(self._finish_seek)
        self.play_button.clicked.connect(self.start_playback)
        self.pause_button.clicked.connect(self.pause_playback)
        self.stop_button.clicked.connect(self.stop_playback)
        self.filter_button.clicked.connect(self.apply_playback_filter)

    def _set_controls_enabled(self, enabled):
        for widget in (
            self.fft_box,
            self.hop_box,
            self.channel_box,
            self.min_freq_edit,
            self.max_freq_edit,
            self.apply_button,
            self.dynamic_slider,
            self.range_start_edit,
            self.range_end_edit,
            self.show_range_button,
            self.full_range_button,
            self.select_range_button,
            self.seek_slider,
            self.play_button,
            self.pause_button,
            self.stop_button,
            self.low_cut_edit,
            self.high_cut_edit,
            self.filter_button,
        ):
            widget.setEnabled(enabled)
        self.export_button.setEnabled(
            enabled and self._displayed_spectrogram is not None
        )
        self.pause_button.setEnabled(False)

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Supported audio (*.wav *.flac *.ogg *.aiff *.aif *.au);;All files (*)",
        )
        if not path:
            return
        self._pending_open_path = path
        if self._playback_worker is not None:
            self._is_playing = False
            self._resume_after_worker = False
            self._pending_seek_sample = 0
            self._playback_worker.request_stop()
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            return
        self._load_file(path)

    def _load_file(self, path):
        try:
            new_file = sf.SoundFile(path, mode="r")
        except Exception as exc:
            self._pending_open_path = None
            QtWidgets.QMessageBox.critical(
                self,
                "Cannot open audio",
                "This file could not be opened by SoundFile/libsndfile. "
                "Try WAV, FLAC or OGG, or convert the file to WAV.\n\n"
                f"Details: {exc}",
            )
            return

        if self.audio_file is not None:
            self.audio_file.close()
        self.audio_file = new_file
        self.file_path = path
        self.sample_rate = int(new_file.samplerate)
        self.total_samples = int(len(new_file))
        self.channel_count = int(new_file.channels)
        self.duration = self.total_samples / self.sample_rate
        self._displayed_spectrogram = None
        self.export_button.setEnabled(False)
        self._playback_position_samples = 0
        self._pending_open_path = None
        self.range_start_edit.setText("0")
        self.range_end_edit.setText(f"{self.duration:.6f}")
        nyquist = self.sample_rate / 2.0
        self.low_cut_edit.setText("0")
        self.high_cut_edit.setText(str(int(min(10000, nyquist))))
        if self._current_min_frequency() >= self._current_max_frequency():
            self.min_freq_edit.setText("0")
        self.seek_slider.setRange(0, max(1, round(self.duration * 1000)))
        self._set_controls_enabled(True)
        self.pause_button.setEnabled(False)
        self.playhead.setMovable(True)
        self.playhead.setValue(0)
        self.playhead.setVisible(True)
        self.plot.setTitle(path.rsplit("/", 1)[-1])
        self.view_box.setRange(
            xRange=(0, max(self.duration, 1.0 / self.sample_rate)),
            yRange=(
                min(self._current_min_frequency(), nyquist),
                min(self._current_max_frequency(), nyquist),
            ),
            padding=0,
        )
        self._update_time_label(0)
        self._refresh_visible_range()

    def _current_max_frequency(self):
        try:
            requested = int(self.max_freq_edit.text())
        except (TypeError, ValueError):
            requested = 10000
        if requested <= 0:
            requested = 10000
        return float(min(requested, self.sample_rate / 2))

    def _current_min_frequency(self):
        try:
            requested = float(self.min_freq_edit.text())
        except (TypeError, ValueError):
            requested = 0.0
        if not math.isfinite(requested) or requested < 0:
            requested = 0.0
        return float(min(requested, self.sample_rate / 2))

    def apply_settings(self):
        if self.audio_file is None:
            return
        try:
            fft_size = int(self.fft_box.currentText())
            hop_length = int(self.hop_box.currentText())
            min_freq = float(self.min_freq_edit.text())
            max_freq = int(self.max_freq_edit.text())
            if fft_size not in self.FFT_CHOICES:
                raise ValueError("Unsupported FFT size")
            if hop_length not in self.HOP_CHOICES:
                raise ValueError("Unsupported hop length")
            if max_freq <= 0:
                raise ValueError("Maximum frequency must be positive")
            nyquist = self.sample_rate / 2
            if not math.isfinite(min_freq) or min_freq < 0:
                raise ValueError("Minimum frequency must be a non-negative number")
            if min_freq >= min(max_freq, nyquist):
                raise ValueError(
                    "Minimum frequency must be lower than the maximum "
                    f"frequency (Nyquist: {nyquist:g} Hz)"
                )
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        if hop_length > fft_size:
            hop_length = fft_size
            self.hop_box.setCurrentText(str(hop_length))
        self.view_box.setYRange(
            min_freq, min(max_freq, self.sample_rate / 2), padding=0
        )
        self._refresh_visible_range()

    def on_dynamic_change(self, value):
        self.dynamic_label.setText(f"{value} dB")
        self.image.setLevels((-float(value), 0.0))
        self.color_bar.setLevels(values=(-float(value), 0.0))

    def _set_selection_mode(self, enabled):
        self.view_box.selection_enabled = enabled
        self.plot.setCursor(
            QtCore.Qt.CursorShape.CrossCursor
            if enabled
            else QtCore.Qt.CursorShape.ArrowCursor
        )

    def _on_range_selected(self, start, end, finished):
        low, high = sorted((start, end))
        if self._selection_region is None:
            self._selection_region = pg.LinearRegionItem(
                values=(low, high),
                movable=False,
                brush=pg.mkBrush(80, 150, 255, 45),
                pen=pg.mkPen(100, 170, 255, 150),
            )
            self.plot.addItem(self._selection_region)
        else:
            self._selection_region.setRegion((low, high))

        if finished:
            if high - low <= 1.0 / max(1, self.sample_rate):
                self.plot.removeItem(self._selection_region)
                self._selection_region = None
            else:
                low = max(0.0, low)
                high = min(self.duration, high)
                if high > low:
                    self.range_start_edit.setText(f"{low:.6f}")
                    self.range_end_edit.setText(f"{high:.6f}")
                    self.view_box.setXRange(low, high, padding=0)
                    self._refresh_visible_range()
                self.plot.removeItem(self._selection_region)
                self._selection_region = None
            self.select_range_button.setChecked(False)

    def _on_x_range_changed(self, *args):
        if self.audio_file is None or self.view_box.selection_enabled:
            return
        low, high = self.view_box.viewRange()[0]
        low = max(0.0, min(self.duration, low))
        high = max(0.0, min(self.duration, high))
        if high > low:
            self.range_start_edit.setText(f"{low:.6f}")
            self.range_end_edit.setText(f"{high:.6f}")
            self._render_timer.start()

    def show_time_range(self):
        if self.audio_file is None:
            return
        try:
            start = float(self.range_start_edit.text().strip())
            end = float(self.range_end_edit.text().strip())
            if not (math.isfinite(start) and math.isfinite(end)):
                raise ValueError("Time values must be finite numbers.")
            if start < 0 or end > self.duration or start >= end:
                raise ValueError(
                    f"Enter 0 <= From < To <= {self.duration:.6f} seconds."
                )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid time range", str(exc))
            return
        self.view_box.setXRange(start, end, padding=0)
        self._refresh_visible_range()

    def show_full_recording(self):
        if self.audio_file is None:
            return
        self.range_start_edit.setText("0")
        self.range_end_edit.setText(f"{self.duration:.6f}")
        self.view_box.setXRange(
            0, max(self.duration, 1.0 / self.sample_rate), padding=0
        )
        self._refresh_visible_range()

    def _refresh_visible_range(self):
        if self.audio_file is None:
            return
        low, high = self.view_box.viewRange()[0]
        start = max(0.0, min(self.duration, low))
        end = max(0.0, min(self.duration, high))
        if end <= start:
            self.status.setText("The selected region is outside the audio file.")
            return
        if self._render_cancel is not None:
            self._render_cancel.set()
        if self._render_future is not None:
            self._render_future.cancel()
        self._render_generation += 1
        generation = self._render_generation
        cancel_event = threading.Event()
        self._render_cancel = cancel_event
        self.status.setText("Calculating visible spectrogram…")
        self._render_future = self._render_executor.submit(
            calculate_visible_spectrogram,
            self.file_path,
            self.sample_rate,
            self.total_samples,
            start,
            end,
            int(self.fft_box.currentText()),
            min(int(self.hop_box.currentText()), int(self.fft_box.currentText())),
            self._current_min_frequency(),
            self._current_max_frequency(),
            self.channel_box.currentText(),
            cancel_event,
        )
        QtCore.QTimer.singleShot(
            30, lambda: self._poll_render(generation, start, end)
        )

    def _poll_render(self, generation, start, end):
        future = self._render_future
        if generation != self._render_generation or future is None:
            return
        if not future.done():
            QtCore.QTimer.singleShot(
                30, lambda: self._poll_render(generation, start, end)
            )
            return
        try:
            result = future.result()
        except InterruptedError:
            return
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Spectrogram error", str(exc))
            return
        if result is None:
            return

        matrix, time_edges, frequency_edges, effective_hop = result
        if generation != self._render_generation:
            return
        self.image.setImage(
            matrix,
            autoLevels=False,
            levels=(-float(self.dynamic_slider.value()), 0.0),
        )
        rect = QtCore.QRectF(
            float(time_edges[0]),
            float(frequency_edges[0]),
            float(time_edges[-1] - time_edges[0]),
            float(frequency_edges[-1] - frequency_edges[0]),
        )
        self.image.setRect(rect)
        self.image.setLevels((-float(self.dynamic_slider.value()), 0.0))
        self.color_bar.setLevels(
            values=(-float(self.dynamic_slider.value()), 0.0)
        )
        self._displayed_spectrogram = {
            "spectrogram_db": matrix.copy(),
            "time_edges": time_edges.copy(),
            "frequency_edges": frequency_edges.copy(),
            "metadata": {
                "source_file": self.file_path.rsplit("/", 1)[-1],
                "sample_rate": self.sample_rate,
                "fft_size": int(self.fft_box.currentText()),
                "hop_length": effective_hop,
                "channel": self.channel_box.currentText(),
                "dynamic_range_db": self.dynamic_slider.value(),
                "requested_time_range": (start, end),
                "displayed_time_range": (
                    float(time_edges[0]),
                    float(time_edges[-1]),
                ),
                "displayed_frequency_range": (
                    float(frequency_edges[0]),
                    float(frequency_edges[-1]),
                ),
            },
        }
        self.export_button.setEnabled(True)
        window_ms = int(self.fft_box.currentText()) / self.sample_rate * 1000
        hop_ms = effective_hop / self.sample_rate * 1000
        self.status.setText(
            f"{start:.3f}–{end:.3f} s | "
            f"{matrix.shape[1]} time columns × {matrix.shape[0]} frequency rows | "
            f"FFT window {window_ms:.1f} ms | frame step {hop_ms:.3f} ms"
        )

    def export_visible_spectrogram(self):
        if self._displayed_spectrogram is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export visible spectrogram data",
            "spectrogram_segment.pkl",
            "Pickle files (*.pkl)",
        )
        if not path:
            return
        if not path.lower().endswith(".pkl"):
            path += ".pkl"
        try:
            with open(path, "wb") as output_file:
                pickle.dump(
                    self._displayed_spectrogram,
                    output_file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except (OSError, pickle.PickleError, TypeError) as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not export spectrogram data:\n{exc}"
            )
            return
        self.status.setText(f"Exported spectrogram data to {path}")

    def _get_playback_filter(self, show_error=True):
        try:
            return build_playback_filter(
                float(self.low_cut_edit.text().strip()),
                float(self.high_cut_edit.text().strip()),
                self.sample_rate,
            )
        except (TypeError, ValueError) as exc:
            if show_error:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid playback filter", str(exc)
                )
            return False

    def apply_playback_filter(self):
        if self.audio_file is None:
            return
        filter_sos = self._get_playback_filter()
        if filter_sos is False:
            return
        self._filter_sos = filter_sos
        if self._is_playing:
            self._resume_after_worker = True
            self._stop_worker()
        low = float(self.low_cut_edit.text())
        high = float(self.high_cut_edit.text())
        self.status.setText(
            f"Playback filter: {low:g}–{high:g} Hz. "
            "The spectrogram itself is unchanged."
        )

    def start_playback(self):
        if self.audio_file is None:
            return
        if sd is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Playback dependency missing",
                "Install the playback library first:\n\n"
                "python -m pip install sounddevice",
            )
            return
        if self._playback_worker is not None:
            if self._is_playing:
                return
            self._resume_after_worker = True
            return
        filter_sos = self._get_playback_filter()
        if filter_sos is False:
            return
        self._filter_sos = filter_sos
        if self._playback_position_samples >= self.total_samples:
            self._playback_position_samples = 0
        self._start_worker()

    def _start_worker(self):
        worker = PlaybackWorker(
            self.file_path,
            self.sample_rate,
            self._playback_position_samples,
            self.channel_box.currentText(),
            self._filter_sos,
            self,
        )
        worker.finished.connect(self._on_worker_finished)
        self._playback_worker = worker
        self._is_playing = True
        self._resume_after_worker = False
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        worker.start()
        self._playback_timer.start()

    def _stop_worker(self):
        if self._playback_worker is None:
            return
        self._is_playing = False
        self._playback_worker.request_stop()
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)

    def pause_playback(self):
        if not self._is_playing:
            return
        self._resume_after_worker = False
        self._stop_worker()

    def stop_playback(self):
        self._resume_after_worker = False
        if self._playback_worker is not None:
            self._pending_seek_sample = 0
            self._stop_worker()
            return
        self._set_playback_position(0)

    def _on_worker_finished(self):
        worker = self.sender()
        if worker is not self._playback_worker:
            return
        self._playback_position_samples = min(
            self.total_samples, worker.position_samples
        )
        self._playback_worker = None
        was_playing = self._is_playing
        self._is_playing = False

        if self._pending_seek_sample is not None:
            self._playback_position_samples = self._pending_seek_sample
            self._pending_seek_sample = None
        elif was_playing and worker.error is None:
            self._playback_position_samples = self.total_samples
        if not self._seeking:
            self._set_playback_position(self._playback_position_samples)

        if worker.error:
            QtWidgets.QMessageBox.critical(self, "Playback error", worker.error)
        should_resume = self._resume_after_worker and not self._seeking
        if not self._seeking:
            self._resume_after_worker = False
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self._playback_timer.stop()
        self.stop_button.setEnabled(True)

        pending_open_path = self._pending_open_path
        if pending_open_path is not None:
            self._load_file(pending_open_path)
        elif should_resume:
            self.start_playback()
        worker.deleteLater()

    def _update_playback_position(self):
        worker = self._playback_worker
        if worker is None or not self._is_playing or self._seeking:
            return
        self._set_playback_position(worker.position_samples)

    def _set_playback_position(self, sample_position):
        self._playback_position_samples = min(
            self.total_samples, max(0, int(sample_position))
        )
        seconds = self._playback_position_samples / max(1, self.sample_rate)
        self._syncing_controls = True
        try:
            self.seek_slider.setValue(round(seconds * 1000))
            self.playhead.setValue(seconds)
        finally:
            self._syncing_controls = False
        self._update_time_label(self._playback_position_samples)

    def _update_time_label(self, sample_position):
        current = sample_position / max(1, self.sample_rate)
        self.time_label.setText(
            f"{format_time(current)} / {format_time(self.duration)}"
        )

    def _begin_seek(self):
        if self.audio_file is None:
            return
        if not self._seeking:
            self._seeking = True
            self._resume_after_worker = self._is_playing
            if self._is_playing:
                self._stop_worker()

    def _preview_seek(self, milliseconds):
        if self._syncing_controls or self.audio_file is None:
            return
        seconds = milliseconds / 1000.0
        self._syncing_controls = True
        try:
            self.playhead.setValue(seconds)
        finally:
            self._syncing_controls = False
        self.time_label.setText(
            f"{format_time(seconds)} / {format_time(self.duration)}"
        )

    def _on_playhead_changed(self):
        if self._syncing_controls or self.audio_file is None:
            return
        if not self._seeking:
            self._seeking = True
            self._resume_after_worker = self._is_playing
            if self._is_playing:
                self._stop_worker()
        seconds = min(self.duration, max(0.0, self.playhead.value()))
        self._syncing_controls = True
        try:
            self.seek_slider.setValue(round(seconds * 1000))
        finally:
            self._syncing_controls = False
        self.time_label.setText(
            f"{format_time(seconds)} / {format_time(self.duration)}"
        )

    def _finish_seek(self):
        if not self._seeking or self.audio_file is None:
            return
        seconds = min(self.duration, max(0.0, self.playhead.value()))
        self._pending_seek_sample = min(
            self.total_samples, round(seconds * self.sample_rate)
        )
        self._playback_position_samples = self._pending_seek_sample
        resume = self._resume_after_worker
        self._seeking = False
        if self._playback_worker is not None:
            self._resume_after_worker = resume
        else:
            self._pending_seek_sample = None
            self._set_playback_position(self._playback_position_samples)
            self._resume_after_worker = False
            if resume:
                self.start_playback()

    def closeEvent(self, event):
        self._render_timer.stop()
        if self._render_cancel is not None:
            self._render_cancel.set()
        self._render_executor.shutdown(wait=False, cancel_futures=True)
        if self._playback_worker is not None:
            self._playback_worker.request_stop()
            self._playback_worker.wait(2000)
        if self.audio_file is not None:
            self.audio_file.close()
            self.audio_file = None
        event.accept()
