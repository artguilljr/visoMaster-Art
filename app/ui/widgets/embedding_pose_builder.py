"""Build a VisoMaster embedding from a small set of aligned pose references."""

from dataclasses import dataclass, replace
from math import cos, radians, sin
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


POSES = ("Front", "Turn left", "Turn right", "Look up", "Look down")
SLOT_MIME = "application/x-visomaster-pose-slot"
GALLERY_MIME = "application/x-visomaster-gallery-face"


@dataclass
class PosePhoto:
    name: str
    rgb: np.ndarray
    embeddings: dict
    origin: str = ""
    scale: int = 100

    def display_rgb(self):
        if self.scale == 100:
            return self.rgb
        image = image_from_rgb(self.rgb)
        size = round(224 * self.scale / 100)
        scaled = image.scaled(
            size, size, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        canvas = QtGui.QImage(224, 224, QtGui.QImage.Format.Format_RGB888)
        canvas.fill(QtGui.QColor(32, 32, 32))
        with QtGui.QPainter(canvas) as painter:
            offset = (224 - size) // 2
            painter.drawImage(offset, offset, scaled)
        return aligned_rgb(canvas)


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
    gallery_dropped = QtCore.Signal(str, int)

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
        if mime.hasFormat(SLOT_MIME) or mime.hasFormat(GALLERY_MIME) or (
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
        elif mime.hasFormat(GALLERY_MIME):
            index = bytes(mime.data(GALLERY_MIME))
            if index.isdigit():
                self.gallery_dropped.emit(self.pose, int(index))
                event.acceptProposedAction()
        elif len(mime.urls()) == 1 and mime.urls()[0].isLocalFile():
            self.file_dropped.emit(self.pose, mime.urls()[0].toLocalFile())
            event.acceptProposedAction()


class FaceGallery(QtWidgets.QListWidget):
    def __init__(self):
        super().__init__()
        self.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.setIconSize(QtCore.QSize(64, 64))
        self.setGridSize(QtCore.QSize(84, 82))
        self.setDragEnabled(True)
        self.setMinimumHeight(118)
        self.setAccessibleName("Detected faces from folder")

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if item is None:
            return
        mime = QtCore.QMimeData()
        mime.setData(GALLERY_MIME, str(self.row(item)).encode())
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(item.icon().pixmap(self.iconSize()))
        drag.exec(QtCore.Qt.DropAction.CopyAction)


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
        self.gallery_photos = []
        self.scale_gallery_index = None

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
            slot.gallery_dropped.connect(self.place_gallery_face)
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
        gallery_box = QtWidgets.QGroupBox("Folder faces")
        gallery_layout = QtWidgets.QVBoxLayout(gallery_box)
        folder_row = QtWidgets.QHBoxLayout()
        self.folder_button = QtWidgets.QPushButton("Choose folder")
        self.folder_button.clicked.connect(self.choose_folder)
        folder_row.addWidget(self.folder_button)
        self.folder_status = QtWidgets.QLabel("Drag a detected face into a pose slot.")
        self.folder_status.setWordWrap(True)
        folder_row.addWidget(self.folder_status, 1)
        gallery_layout.addLayout(folder_row)
        self.gallery = FaceGallery()
        self.gallery.currentRowChanged.connect(self.select_gallery_face)
        self.gallery.itemDoubleClicked.connect(
            lambda item: self.place_gallery_face(self.selected_pose, self.gallery.row(item))
        )
        gallery_layout.addWidget(self.gallery)
        scale_row = QtWidgets.QHBoxLayout()
        self.scale_preview = QtWidgets.QLabel()
        self.scale_preview.setFixedSize(64, 64)
        scale_row.addWidget(self.scale_preview)
        self.scale_label = QtWidgets.QLabel("Crop scale")
        scale_row.addWidget(self.scale_label)
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.scale_slider.setRange(50, 200)
        self.scale_slider.setAccessibleName("Selected face crop scale")
        self.scale_slider.setToolTip("Scale the crop in the composite preview")
        self.scale_slider.valueChanged.connect(self.set_crop_scale)
        scale_row.addWidget(self.scale_slider, 1)
        gallery_layout.addLayout(scale_row)
        board_layout.addWidget(gallery_box, 1)
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
        self.scale_gallery_index = None
        self.refresh_scale()

    def select_gallery_face(self, index):
        self.scale_gallery_index = index if index >= 0 else None
        self.refresh_scale()

    def refresh_scale(self):
        index = self.scale_gallery_index
        photo = self.photos.get(self.selected_pose) if index is None else self.gallery_photos[index]
        with QtCore.QSignalBlocker(self.scale_slider):
            self.scale_slider.setEnabled(photo is not None)
            self.scale_slider.setValue(photo.scale if photo else 100)
        self.scale_label.setText(f"Crop scale {photo.scale}%" if photo else "Crop scale")
        if photo:
            self.scale_preview.setPixmap(self.face_icon(photo).pixmap(64, 64))
        else:
            self.scale_preview.clear()

    def set_crop_scale(self, value):
        index = self.scale_gallery_index
        if index is None:
            photo = self.photos.get(self.selected_pose)
            if photo:
                self.photos[self.selected_pose] = replace(photo, scale=value)
        else:
            photo = replace(self.gallery_photos[index], scale=value)
            self.gallery_photos[index] = photo
            self.gallery.item(index).setIcon(self.face_icon(photo))
        self.refresh()

    @staticmethod
    def face_icon(photo):
        return QtGui.QIcon(QtGui.QPixmap.fromImage(image_from_rgb(photo.display_rgb())))

    def place_gallery_face(self, pose, index):
        if 0 <= index < len(self.gallery_photos):
            self.photos[pose] = self.gallery_photos[index]
            self.slots[pose].setChecked(True)
            self.select_pose(pose)
            self.refresh()

    def choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose face photo folder")
        if not folder:
            return
        from app.helpers.miscellaneous import get_image_files

        files = sorted(str(Path(folder) / name) for name in get_image_files(folder))
        if not files:
            self.folder_status.setText("No supported photos in this folder.")
            return
        self.start_photo_loader(files, folder=folder)

    def choose_photo(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose pose photo", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
        )
        if path:
            self.load_photo(self.selected_pose, path)

    def load_photo(self, pose, path):
        self.start_photo_loader([path], pose=pose)

    def start_photo_loader(self, files, *, pose=None, folder=None):
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

        if folder:
            self.gallery.clear()
            self.gallery_photos.clear()
            self.folder_status.setText(f"Scanning {len(files)} photos in {Path(folder).name}...")
        else:
            self.status.setText(f"Finding a face in {Path(files[0]).name}...")
        self.folder_button.setEnabled(False)
        self.save.setEnabled(False)
        self.loaded_photo = None
        self.pending_photo = pose, files[0]
        self.loader = PosePhotoLoader(
            self.main_window, files_list=files, parent=self,
            max_faces_per_image=0 if folder else 1,
        )
        self.loader.thumbnail_ready.connect(self.photo_loaded)
        self.loader.finished.connect(self.finish_loading, QtCore.Qt.ConnectionType.QueuedConnection)
        self.loader.start()

    @QtCore.Slot(str, np.ndarray, object, QtGui.QImage, str)
    def photo_loaded(self, path, _crop, embeddings, image, _face_id):
        photo = PosePhoto(Path(path).name, aligned_rgb(image), embeddings)
        if self.pending_photo[0] is None:
            self.gallery_photos.append(photo)
            item = QtWidgets.QListWidgetItem(self.face_icon(photo), "")
            item.setToolTip(photo.name)
            item.setData(QtCore.Qt.ItemDataRole.AccessibleTextRole, photo.name)
            self.gallery.addItem(item)
        else:
            self.loaded_photo = photo

    @QtCore.Slot()
    def finish_loading(self):
        if self.loader is None:
            return
        pose, path = self.pending_photo
        self.loader.wait()
        self.loader.deleteLater()
        self.loader = None
        self.folder_button.setEnabled(True)
        if pose is None:
            count = len(self.gallery_photos)
            self.folder_status.setText(
                f"{count} detected face(s). Drag into a slot." if count else "No usable faces found."
            )
            self.refresh()
            return
        if self.loaded_photo is None:
            self.refresh()
            self.status.setText(f"No usable face found in {Path(path).name}.")
            return
        self.photos[pose] = self.loaded_photo
        self.slots[pose].setChecked(True)
        self.select_pose(pose)
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
                icon = self.face_icon(photo)
            slot.setIcon(icon)
        self.refresh_scale()
        samples = [photo.display_rgb() for photo in self.photos.values()]
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
