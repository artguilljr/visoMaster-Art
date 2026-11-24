"""Build a VisoMaster embedding from a small set of aligned pose references."""

from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


POSES = ("Front", "Turn left", "Turn right", "Look up", "Look down")
SLOT_MIME = "application/x-visomaster-pose-slot"
GALLERY_MIME = "application/x-visomaster-gallery-face"


@dataclass(eq=False)
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
    files_dropped = QtCore.Signal(str, list)
    slot_dropped = QtCore.Signal(str, str)
    gallery_dropped = QtCore.Signal(str, list)
    review_requested = QtCore.Signal(str)

    def __init__(self, pose):
        super().__init__(pose)
        self.pose = pose
        self.start = None
        self.setAcceptDrops(True)
        self.setCheckable(True)
        self.setMinimumSize(122, 76)
        self.setIconSize(QtCore.QSize(46, 46))

    def contextMenuEvent(self, event):
        self.review_requested.emit(self.pose)
        event.accept()

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
            mime.hasUrls() and all(url.isLocalFile() for url in mime.urls())
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
            indexes = bytes(mime.data(GALLERY_MIME)).split(b",")
            if all(index.isdigit() for index in indexes):
                self.gallery_dropped.emit(self.pose, [int(index) for index in indexes])
                event.acceptProposedAction()
        elif mime.hasUrls() and all(url.isLocalFile() for url in mime.urls()):
            self.files_dropped.emit(self.pose, [url.toLocalFile() for url in mime.urls()])
            event.acceptProposedAction()


class FaceGallery(QtWidgets.QListWidget):
    def __init__(self):
        super().__init__()
        self.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.setIconSize(QtCore.QSize(64, 64))
        self.setGridSize(QtCore.QSize(84, 82))
        self.setDragEnabled(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setMinimumHeight(118)
        self.setAccessibleName("Detected faces from folder")

    def startDrag(self, supported_actions):
        items = self.selectedItems()
        if not items:
            return
        mime = QtCore.QMimeData()
        mime.setData(GALLERY_MIME, ",".join(str(self.row(item)) for item in items).encode())
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(items[0].icon().pixmap(self.iconSize()))
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
        self.combined_embedding = None

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
            slot.files_dropped.connect(self.load_photos)
            slot.slot_dropped.connect(self.move_photo)
            slot.gallery_dropped.connect(self.place_gallery_faces)
            slot.review_requested.connect(self.review_slot)
            grid.addWidget(slot, row, column)
        self.slots["Front"].setChecked(True)
        board_layout.addLayout(grid)

        actions = QtWidgets.QHBoxLayout()
        for caption, action in (
            ("Choose photos", self.choose_photos),
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
        self.gallery.itemDoubleClicked.connect(
            lambda item: self.place_gallery_faces(self.selected_pose, [self.gallery.row(item)])
        )
        gallery_layout.addWidget(self.gallery)
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

    @staticmethod
    def face_icon(photo):
        return QtGui.QIcon(QtGui.QPixmap.fromImage(image_from_rgb(photo.rgb)))

    def add_pose_photos(self, pose, photos):
        current = self.photos.setdefault(pose, [])
        for photo in photos:
            if photo in current:
                continue
            if not photo.origin:
                current[:] = [existing for existing in current if not existing.origin]
            current.append(photo)

    def place_gallery_faces(self, pose, indexes):
        self.add_pose_photos(
            pose, [self.gallery_photos[index] for index in indexes
                   if 0 <= index < len(self.gallery_photos)]
        )
        self.slots[pose].setChecked(True)
        self.select_pose(pose)
        self.refresh()

    def review_slot(self, pose):
        if self.loader is not None:
            return
        photos = self.photos.get(pose)
        if not photos:
            self.status.setText(f"{pose} has no references to review.")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"{pose} references")
        dialog.resize(540, 360)
        layout = QtWidgets.QVBoxLayout(dialog)
        content = QtWidgets.QHBoxLayout()
        entries = QtWidgets.QListWidget()
        entries.setIconSize(QtCore.QSize(48, 48))
        content.addWidget(entries)
        preview = QtWidgets.QLabel()
        preview.setFixedSize(224, 224)
        content.addWidget(preview)
        layout.addLayout(content)
        controls = QtWidgets.QHBoxLayout()
        destination = QtWidgets.QComboBox()
        destination.setAccessibleName("Move selected face to pose")
        destination.addItems([candidate for candidate in POSES if candidate != pose])
        controls.addWidget(destination)
        move = QtWidgets.QPushButton("Move to pose")
        remove = QtWidgets.QPushButton("Remove face")
        close = QtWidgets.QPushButton("Close")
        for button in (move, remove, close):
            controls.addWidget(button)
        layout.addLayout(controls)

        def show_photo(index):
            selected = 0 <= index < len(photos)
            move.setEnabled(selected)
            remove.setEnabled(selected)
            if selected:
                preview.setPixmap(self.face_icon(photos[index]).pixmap(224, 224))
            else:
                preview.clear()

        def populate(index=0):
            entries.clear()
            for photo in photos:
                entries.addItem(QtWidgets.QListWidgetItem(
                    self.face_icon(photo), photo.name or photo.origin
                ))
            entries.setCurrentRow(min(index, len(photos) - 1))
            show_photo(entries.currentRow())

        def update_photo(move_to=None):
            index = entries.currentRow()
            if index < 0:
                return
            photo = photos.pop(index)
            if move_to is not None:
                self.add_pose_photos(move_to, [photo])
            self.refresh()
            populate(index)

        entries.currentRowChanged.connect(show_photo)
        move.clicked.connect(lambda: update_photo(destination.currentText()))
        remove.clicked.connect(lambda: update_photo())
        close.clicked.connect(dialog.accept)
        populate()
        dialog.exec()

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

    def choose_photos(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Choose pose photos", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
        )
        if paths:
            self.load_photos(self.selected_pose, paths)

    def load_photos(self, pose, paths):
        self.start_photo_loader(paths, pose=pose)

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
            self.status.setText(f"Finding faces in {len(files)} selected photo(s)...")
        self.folder_button.setEnabled(False)
        self.save.setEnabled(False)
        self.loaded_photo_count = 0
        self.pending_pose = pose
        self.loader = PosePhotoLoader(
            self.main_window, files_list=files, parent=self,
            max_faces_per_image=0 if folder else 1, align_preview=True,
        )
        self.loader.thumbnail_ready.connect(self.photo_loaded)
        self.loader.finished.connect(self.finish_loading, QtCore.Qt.ConnectionType.QueuedConnection)
        self.loader.start()

    @QtCore.Slot(str, np.ndarray, object, QtGui.QImage, str)
    def photo_loaded(self, path, _crop, embeddings, image, _face_id):
        photo = PosePhoto(Path(path).name, aligned_rgb(image), embeddings)
        if self.pending_pose is None:
            self.gallery_photos.append(photo)
            item = QtWidgets.QListWidgetItem(self.face_icon(photo), "")
            item.setToolTip(photo.name)
            item.setData(QtCore.Qt.ItemDataRole.AccessibleTextRole, photo.name)
            self.gallery.addItem(item)
        else:
            self.loaded_photo_count += 1
            self.add_pose_photos(self.pending_pose, [photo])
            self.refresh()

    @QtCore.Slot()
    def finish_loading(self):
        if self.loader is None:
            return
        pose = self.pending_pose
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
        if not self.loaded_photo_count:
            self.refresh()
            self.status.setText("No usable faces found in the selected photos.")
            return
        self.slots[pose].setChecked(True)
        self.select_pose(pose)
        self.refresh()

    def clear_slot(self):
        if self.loader is None:
            self.photos.pop(self.selected_pose, None)
            self.refresh()

    def move_photo(self, source, target):
        if self.loader is not None or source == target or not self.photos.get(source):
            return
        self.add_pose_photos(target, self.photos.pop(source))
        self.refresh()

    def mirror_side(self):
        if self.loader is not None:
            return
        for source, target in (("Turn left", "Turn right"), ("Turn right", "Turn left")):
            real = [photo.rgb for photo in self.photos.get(source, []) if not photo.origin]
            existing = self.photos.get(target, [])
            if real and all(photo.origin == "Estimated" for photo in existing):
                rgb = np.rint(reduce_samples(real, self.method.currentText())).astype(np.uint8)
                self.photos[target] = [PosePhoto("", rgb[:, ::-1].copy(), {}, "Mirrored")]
                self.refresh()
                return
        self.status.setText("Add a real side photo and leave the opposite side empty.")

    def fill_missing(self):
        if self.loader is not None:
            return
        real = [photo.rgb for slot in self.photos.values() for photo in slot if not photo.origin]
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
            if self.photos.get(pose):
                continue
            samples = [
                photo.rgb for key in adjacent for photo in self.photos.get(key, [])
                if not photo.origin
            ]
            rgb = reduce_samples(samples if len(samples) >= 2 else real, "Mean")
            self.photos[pose] = [PosePhoto("", np.rint(rgb).astype(np.uint8), {}, "Estimated")]
        self.refresh()

    def refresh(self):
        method = self.method.currentText()
        for pose, slot in self.slots.items():
            photos = self.photos.get(pose, [])
            detail = ""
            icon = QtGui.QIcon()
            if photos:
                detail = photos[0].origin or f"{len(photos)} photo(s)"
                rgb = np.rint(reduce_samples([photo.rgb for photo in photos], method))
                icon = QtGui.QIcon(QtGui.QPixmap.fromImage(image_from_rgb(rgb)))
            slot.setText(f"{pose}\n{detail}" if detail else pose)
            slot.setToolTip("Right-click to review or move individual faces" if photos else "Drop photos here")
            slot.setIcon(icon)
        photos = list(dict.fromkeys(photo for slot in self.photos.values() for photo in slot))
        self.preview.image = (
            image_from_rgb(np.rint(reduce_samples([p.rgb for p in photos], method)))
            if photos else QtGui.QImage()
        )
        self.preview.update()
        real = [photo for photo in photos if not photo.origin]
        self.combined_embedding = None
        message = "Drop photos on a slot or select a slot and choose photos."
        if real:
            try:
                self.combined_embedding = merge_embeddings(real, method)
                message = f"{len(real)} real reference(s) merged. Generated views are preview only."
            except ValueError as error:
                message = str(error)
        if self.loader is None:
            self.status.setText(message)
        self.save.setEnabled(self.loader is None and self.combined_embedding is not None)

    def save_embedding(self):
        if self.combined_embedding is None:
            return
        name, accepted = QtWidgets.QInputDialog.getText(self, "Add Embedding", "Name")
        if accepted and name.strip():
            self.add_embedding(name.strip(), self.combined_embedding)
