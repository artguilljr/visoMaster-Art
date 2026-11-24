"""Build a VisoMaster embedding from a small set of aligned pose references."""

from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


POSES = ("Front", "Turn left", "Turn right", "Look up", "Look down")
SLOT_MIME = "application/x-visomaster-pose-slot"


@dataclass
class PosePhoto:
    name: str
    rgb: np.ndarray
    embeddings: dict
    origin: str = ""


def reduce_samples(samples, method):
    reducer = np.median if method == "Median" else np.mean
    return reducer(np.stack(samples), axis=0)


def merge_embeddings(photos, method):
    if not photos:
        raise ValueError("Add at least one real photo.")
    shared = set.intersection(*(set(photo.embeddings) for photo in photos))
    models = sorted(shared - {"kps_5"})
    if not models:
        raise ValueError("Load all contributing photos with the same recognition model.")
    result = {}
    for model in models:
        samples = [np.asarray(photo.embeddings[model], dtype=np.float64) for photo in photos]
        merged = reduce_samples(samples, method)
        norm = np.linalg.norm(merged)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"The combined embedding for {model} is invalid.")
        result[model] = (merged / norm).tolist()
    if "kps_5" in shared:
        result["kps_5"] = reduce_samples(
            [photo.embeddings["kps_5"] for photo in photos], "Mean"
        ).tolist()
    return result


def image_from_rgb(rgb):
    pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = pixels.shape[:2]
    return QtGui.QImage(
        pixels.data, width, height, pixels.strides[0], QtGui.QImage.Format.Format_RGB888
    ).copy()


def aligned_rgb(image):
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888).scaled(
        224, 224, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation,
    )
    return np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        224, image.bytesPerLine()
    )[:, :672].reshape(224, 224, 3).copy()


class PoseSlot(QtWidgets.QPushButton):
    file_dropped = QtCore.Signal(str, str)
    slot_dropped = QtCore.Signal(str, str)

    def __init__(self, pose):
        super().__init__(pose)
        self.pose = pose
        self.start = None
        self.setAcceptDrops(True)
        self.setCheckable(True)
        self.setMinimumSize(122, 76)
        self.setIconSize(QtCore.QSize(46, 46))

    def mousePressEvent(self, event):
        self.start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self.start is not None
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
            and not self.icon().isNull()
            and (event.position().toPoint() - self.start).manhattanLength()
            >= QtWidgets.QApplication.startDragDistance()
        ):
            self.start = None
            self.setDown(False)
            mime = QtCore.QMimeData()
            mime.setData(SLOT_MIME, self.pose.encode())
            drag = QtGui.QDrag(self)
            drag.setMimeData(mime)
            drag.setPixmap(self.icon().pixmap(self.iconSize()))
            drag.exec(QtCore.Qt.DropAction.MoveAction)
            return
        super().mouseMoveEvent(event)

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasFormat(SLOT_MIME) or (
            len(mime.urls()) == 1 and mime.urls()[0].isLocalFile()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event):
        mime = event.mimeData()
        if mime.hasFormat(SLOT_MIME):
            source = bytes(mime.data(SLOT_MIME)).decode(errors="replace")
            if source in POSES:
                self.slot_dropped.emit(source, self.pose)
                event.acceptProposedAction()
        elif len(mime.urls()) == 1 and mime.urls()[0].isLocalFile():
            self.file_dropped.emit(self.pose, mime.urls()[0].toLocalFile())
            event.acceptProposedAction()


class CompositePreview(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.image = QtGui.QImage()
        self.yaw = self.pitch = 0.0
        self.drag_position = None
        self.setMinimumSize(290, 290)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setAccessibleName("Rotatable composite preview")

    def reset_view(self):
        self.yaw = self.pitch = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        if self.image.isNull():
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Add a pose photo to preview")
            return
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        yaw, pitch = radians(self.yaw), radians(self.pitch)
        half = min(self.width(), self.height()) * 0.34
        center = QtCore.QPointF(self.rect().center())
        source = QtGui.QPolygonF()
        target = QtGui.QPolygonF()
        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            source.append(QtCore.QPointF((x + 1) * self.image.width() / 2,
                                         (y + 1) * self.image.height() / 2))
            z = -x * sin(yaw)
            depth = y * sin(pitch) + z * cos(pitch)
            scale = half * 4 / (4 - depth)
            target.append(center + QtCore.QPointF(
                scale * x * cos(yaw), scale * (y * cos(pitch) - z * sin(pitch))
            ))
        transform = QtGui.QTransform()
        if QtGui.QTransform.quadToQuad(source, target, transform):
            painter.setTransform(transform)
            painter.drawImage(0, 0, self.image)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.drag_position = event.position()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.drag_position is not None:
            delta = event.position() - self.drag_position
            self.drag_position = event.position()
            self.yaw = max(-75, min(75, self.yaw + delta.x() * 0.5))
            self.pitch = max(-75, min(75, self.pitch - delta.y() * 0.5))
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.drag_position = None
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)


class EmbeddingPoseBuilder(QtWidgets.QWidget):
    def __init__(self, main_window, add_embedding):
        super().__init__()
        self.main_window = main_window
        self.add_embedding = add_embedding
        self.photos = {}
        self.loader = None
        self.selected_pose = "Front"
        self.slots = {}

        layout = QtWidgets.QHBoxLayout(self)
        board = QtWidgets.QGroupBox("Pose photos")
        board_layout = QtWidgets.QVBoxLayout(board)
        grid = QtWidgets.QGridLayout()
        self.slot_group = QtWidgets.QButtonGroup(self)
        for pose, row, column in (
            ("Look up", 0, 1), ("Turn left", 1, 0), ("Front", 1, 1),
            ("Turn right", 1, 2), ("Look down", 2, 1),
        ):
            slot = PoseSlot(pose)
            self.slots[pose] = slot
            self.slot_group.addButton(slot)
            slot.clicked.connect(lambda checked=False, pose=pose: self.select_pose(pose))
            slot.file_dropped.connect(self.load_photo)
            slot.slot_dropped.connect(self.move_photo)
            grid.addWidget(slot, row, column)
        self.slots["Front"].setChecked(True)
        board_layout.addLayout(grid)

        actions = QtWidgets.QHBoxLayout()
        for caption, action in (
            ("Choose photo", self.choose_photo),
            ("Clear slot", self.clear_slot),
            ("Mirror missing side", self.mirror_side),
            ("Fill missing views", self.fill_missing),
        ):
            button = QtWidgets.QPushButton(caption)
            button.clicked.connect(action)
            actions.addWidget(button)
        board_layout.addLayout(actions)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        board_layout.addWidget(self.status)
        board_layout.addStretch()
        layout.addWidget(board)

        panel = QtWidgets.QGroupBox("Composite preview")
        panel_layout = QtWidgets.QVBoxLayout(panel)
        self.preview = CompositePreview()
        panel_layout.addWidget(self.preview, 1)
        panel_layout.addWidget(QtWidgets.QLabel("Drag to rotate the composite."))
        controls = QtWidgets.QHBoxLayout()
        self.method = QtWidgets.QComboBox()
        self.method.setAccessibleName("Composite and embedding merge method")
        self.method.addItems(["Mean", "Median"])
        self.method.currentTextChanged.connect(self.refresh)
        controls.addWidget(self.method)
        reset = QtWidgets.QPushButton("Reset angle")
        reset.clicked.connect(self.preview.reset_view)
        controls.addWidget(reset)
        panel_layout.addLayout(controls)
        self.save = QtWidgets.QPushButton("Add Embedding to Library")
        self.save.clicked.connect(self.save_embedding)
        panel_layout.addWidget(self.save)
        layout.addWidget(panel, 1)
        self.refresh()

    def select_pose(self, pose):
        self.selected_pose = pose

    def choose_photo(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose pose photo", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
        )
        if path:
            self.load_photo(self.selected_pose, path)

    def load_photo(self, pose, path):
        if self.loader is not None:
            return
        if (
            self.main_window.video_processor.processing
            or self.main_window.video_processor.is_processing_segments
            or getattr(self.main_window, "is_batch_processing", False)
        ):
            self.status.setText("Pause playback or finish rendering before loading a pose photo.")
            return
        from app.ui.widgets.ui_workers import InputFacesLoaderWorker

        class PosePhotoLoader(InputFacesLoaderWorker):
            def run(self):
                try:
                    self.load_faces(files_list=self.files_list)
                except Exception:
                    import logging

                    logging.getLogger(__name__).exception("Could not load pose photo")
                finally:
                    self.finished.emit()

        self.status.setText(f"Finding a face in {Path(path).name}…")
        self.save.setEnabled(False)
        self.loaded_photo = None
        self.pending_photo = pose, path
        self.loader = PosePhotoLoader(self.main_window, files_list=[path], parent=self)
        self.loader.thumbnail_ready.connect(self.photo_loaded)
        self.loader.finished.connect(self.finish_loading, QtCore.Qt.ConnectionType.QueuedConnection)
        self.loader.start()

    @QtCore.Slot(str, np.ndarray, object, QtGui.QImage, str)
    def photo_loaded(self, path, _crop, embeddings, image, _face_id):
        self.loaded_photo = PosePhoto(Path(path).name, aligned_rgb(image), embeddings)

    @QtCore.Slot()
    def finish_loading(self):
        if self.loader is None:
            return
        pose, path = self.pending_photo
        self.loader.wait()
        self.loader.deleteLater()
        self.loader = None
        if self.loaded_photo is None:
            self.refresh()
            self.status.setText(f"No usable face found in {Path(path).name}.")
            return
        self.photos[pose] = self.loaded_photo
        self.refresh()

    def clear_slot(self):
        if self.loader is None:
            self.photos.pop(self.selected_pose, None)
            self.refresh()

    def move_photo(self, source, target):
        if self.loader is not None or source == target or source not in self.photos:
            return
        photo = self.photos.pop(source)
        if target in self.photos:
            self.photos[source] = self.photos[target]
        self.photos[target] = photo
        self.refresh()

    def mirror_side(self):
        if self.loader is not None:
            return
        for source, target in (("Turn left", "Turn right"), ("Turn right", "Turn left")):
            photo = self.photos.get(source)
            if photo and not photo.origin and (
                target not in self.photos or self.photos[target].origin == "Estimated"
            ):
                self.photos[target] = PosePhoto(photo.name, photo.rgb[:, ::-1].copy(), {}, "Mirrored")
                self.refresh()
                return
        self.status.setText("Add a real side photo and leave the opposite side empty.")

    def fill_missing(self):
        if self.loader is not None:
            return
        real = [photo.rgb for photo in self.photos.values() if not photo.origin]
        if len(real) < 2:
            self.status.setText("Add at least two real photos to estimate missing views.")
            return
        neighbors = {
            "Front": ("Turn left", "Turn right", "Look up", "Look down"),
            "Turn left": ("Front", "Look up", "Look down"),
            "Turn right": ("Front", "Look up", "Look down"),
            "Look up": ("Front", "Turn left", "Turn right"),
            "Look down": ("Front", "Turn left", "Turn right"),
        }
        for pose, adjacent in neighbors.items():
            if pose in self.photos:
                continue
            samples = [
                self.photos[key].rgb for key in adjacent
                if key in self.photos and not self.photos[key].origin
            ]
            rgb = reduce_samples(samples if len(samples) >= 2 else real, "Mean")
            self.photos[pose] = PosePhoto("", np.rint(rgb).astype(np.uint8), {}, "Estimated")
        self.refresh()

    def refresh(self):
        for pose, slot in self.slots.items():
            photo = self.photos.get(pose)
            suffix = f"\n{photo.origin}" if photo and photo.origin else ""
            slot.setText(pose + suffix)
            slot.setToolTip(photo.name if photo else "Drop a photo here")
            icon = QtGui.QIcon()
            if photo:
                icon = QtGui.QIcon(QtGui.QPixmap.fromImage(image_from_rgb(photo.rgb)))
            slot.setIcon(icon)
        samples = [photo.rgb for photo in self.photos.values()]
        self.preview.image = (
            image_from_rgb(np.rint(reduce_samples(samples, self.method.currentText())))
            if samples else QtGui.QImage()
        )
        self.preview.update()
        if self.loader is None:
            real_count = sum(not photo.origin for photo in self.photos.values())
            self.save.setEnabled(real_count > 0)
            self.status.setText(
                f"{real_count} real reference(s). Mirrored and estimated views are preview only."
                if samples else "Drop a photo on a slot or select a slot and choose a photo."
            )

    def save_embedding(self):
        photos = [photo for photo in self.photos.values() if not photo.origin]
        try:
            embeddings = merge_embeddings(photos, self.method.currentText())
        except ValueError as error:
            self.status.setText(str(error))
            return
        name, accepted = QtWidgets.QInputDialog.getText(self, "Add Embedding", "Name")
        if accepted and name.strip():
            self.add_embedding(name.strip(), embeddings)
