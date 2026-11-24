"""Build a VisoMaster embedding from a small set of aligned pose references."""

from dataclasses import dataclass, field
from math import cos, radians, sin
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.widgets.embedding_face_mesh import fuse_face_texture, quality_flags, render_mesh, usable_mesh


POSES = ("Front", "Turn left", "Turn right", "Look up", "Look down")
SLOT_MIME = "application/x-visomaster-pose-slot"
GALLERY_MIME = "application/x-visomaster-gallery-face"


@dataclass(eq=False)
class PosePhoto:
    name: str
    rgb: np.ndarray
    embeddings: dict
    origin: str = ""
    landmarks: np.ndarray | None = None
    excluded: bool = False
    sharpness: float = field(init=False)
    thumbnail: np.ndarray = field(init=False)

    def __post_init__(self):
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        self.sharpness = float(cv2.Laplacian(gray[55:178, 55:169], cv2.CV_32F).var())
        self.thumbnail = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)


def sharpen_crop(rgb, strength):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 1.15)
    detail[np.abs(detail) < 3] = 0
    adjustment = detail[..., None] * (strength / 100)
    return np.rint(np.clip(rgb + adjustment, 0, 255)).astype(np.uint8)


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
        self.mesh = None
        self.wireframe = False
        self.render_key = None
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
        if self.mesh is not None:
            size = max(1, min(self.width(), self.height(), 480))
            key = (size, self.yaw, self.pitch, self.wireframe)
            if key != self.render_key:
                rgba = np.ascontiguousarray(render_mesh(
                    *self.mesh, self.yaw, self.pitch, size, self.wireframe
                ))
                self.rendered_mesh = QtGui.QImage(
                    rgba.data, size, size, rgba.strides[0],
                    QtGui.QImage.Format.Format_RGBA8888,
                ).copy()
                self.render_key = key
            painter.drawImage(
                (self.width() - size) // 2, (self.height() - size) // 2, self.rendered_mesh
            )
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
        self.texture_cache_key = None
        self.texture_cache = None
        self.load_refresh = QtCore.QTimer(self)
        self.load_refresh.setSingleShot(True)
        self.load_refresh.setInterval(180)
        self.load_refresh.timeout.connect(self.refresh)

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
        self.preview_hint = QtWidgets.QLabel("Drag to rotate the face.")
        self.preview_hint.setWordWrap(True)
        panel_layout.addWidget(self.preview_hint)
        controls = QtWidgets.QHBoxLayout()
        self.preview_mode = QtWidgets.QComboBox()
        self.preview_mode.setAccessibleName("Face preview mode")
        self.preview_mode.addItems(["Textured 3D", "Wireframe", "Enhanced composite", "Mean / Median composite"])
        self.preview_mode.currentTextChanged.connect(self.refresh)
        controls.addWidget(self.preview_mode)
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
        entries.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        content.addWidget(entries)
        preview = QtWidgets.QLabel()
        preview.setFixedSize(224, 224)
        content.addWidget(preview)
        layout.addLayout(content)
        review_options = QtWidgets.QHBoxLayout()
        overlay = QtWidgets.QCheckBox("Show landmarks")
        overlay.setChecked(True)
        review_options.addWidget(overlay)
        exclude = QtWidgets.QCheckBox("Exclude this photo from merge")
        review_options.addWidget(exclude)
        layout.addLayout(review_options)
        quality = QtWidgets.QLabel()
        quality.setWordWrap(True)
        layout.addWidget(quality)
        controls = QtWidgets.QHBoxLayout()
        destination = QtWidgets.QComboBox()
        destination.setAccessibleName("Move selected face to pose")
        destination.addItems([candidate for candidate in POSES if candidate != pose])
        controls.addWidget(destination)
        move = QtWidgets.QPushButton("Move selected to pose")
        flip = QtWidgets.QPushButton("Flip horizontally")
        sharpen = QtWidgets.QPushButton("Sharpen")
        remove = QtWidgets.QPushButton("Remove selected")
        close = QtWidgets.QPushButton("Close")
        for button in (move, flip, sharpen, remove, close):
            controls.addWidget(button)
        layout.addLayout(controls)

        def show_photo():
            index = entries.currentRow()
            has_selection = bool(entries.selectedItems())
            can_edit = has_selection and self.loader is None
            move.setEnabled(can_edit)
            remove.setEnabled(can_edit)
            can_process = can_edit and any(
                not photos[entries.row(item)].origin for item in entries.selectedItems()
            )
            flip.setEnabled(can_process)
            sharpen.setEnabled(can_process)
            photo = photos[index] if 0 <= index < len(photos) else None
            with QtCore.QSignalBlocker(exclude):
                exclude.setEnabled(self.loader is None and photo is not None and not photo.origin)
                exclude.setChecked(bool(photo and photo.excluded))
            if photo is None:
                preview.clear()
                quality.clear()
                return
            image = image_from_rgb(photo.rgb)
            if overlay.isChecked() and usable_mesh(photo):
                with QtGui.QPainter(image) as painter:
                    painter.setPen(QtGui.QPen(QtGui.QColor(80, 230, 198, 210), 1.5))
                    painter.drawPoints(QtGui.QPolygonF([
                        QtCore.QPointF(float(x), float(y)) for x, y in photo.landmarks[:, :2]
                    ]))
            preview.setPixmap(QtGui.QPixmap.fromImage(image))
            if photo.origin:
                quality.setText(f"{photo.origin} reference for preview only.")
            else:
                references = {
                    candidate for slot in self.photos.values() for candidate in slot
                }
                flags = quality_flags(photo, references)
                quality.setText(" | ".join(flags) if flags else "Alignment and sharpness checks passed.")

        def set_excluded(checked):
            index = entries.currentRow()
            if 0 <= index < len(photos) and not photos[index].origin:
                photos[index].excluded = checked
                entries.item(index).setText(entry_label(photos[index]))
                self.refresh()

        def entry_label(photo):
            name = photo.name or photo.origin
            if photo.excluded:
                return f"{name} (excluded)"
            return name

        def populate(index=0):
            entries.clear()
            for photo in photos:
                entries.addItem(QtWidgets.QListWidgetItem(
                    self.face_icon(photo), entry_label(photo)
                ))
            entries.setCurrentRow(min(index, len(photos) - 1))
            show_photo()

        def update_photo(move_to=None):
            indexes = sorted(entries.row(item) for item in entries.selectedItems())
            if not indexes:
                return
            moved = [photos[index] for index in indexes]
            for index in reversed(indexes):
                photos.pop(index)
            if move_to is not None:
                self.add_pose_photos(move_to, moved)
            self.refresh()
            populate(indexes[0])

        def process_selected(transform, operation):
            indexes = sorted(entries.row(item) for item in entries.selectedItems())
            originals = {
                f"{index}.png": photos[index] for index in indexes if not photos[index].origin
            }
            if not originals:
                return

            def show_result():
                if dialog.isVisible():
                    populate(indexes[0])
                    quality.setText(self.status.text())

            overrides = {
                path: np.ascontiguousarray(transform(photo.rgb)) for path, photo in originals.items()
            }
            self.start_photo_loader(
                list(originals), pose=pose, edit_targets=originals,
                image_overrides=overrides, operation=operation, on_finished=show_result,
            )
            show_photo()
            quality.setText(self.status.text())

        def sharpen_selected():
            strength, accepted = QtWidgets.QInputDialog.getInt(
                dialog, "Sharpen photos", "Strength (%)", 35, 1, 100
            )
            if accepted:
                process_selected(lambda rgb: sharpen_crop(rgb, strength), "Sharpened")

        entries.currentRowChanged.connect(show_photo)
        entries.itemSelectionChanged.connect(show_photo)
        overlay.toggled.connect(show_photo)
        exclude.toggled.connect(set_excluded)
        move.clicked.connect(lambda: update_photo(destination.currentText()))
        flip.clicked.connect(lambda: process_selected(lambda rgb: rgb[:, ::-1], "Flipped"))
        sharpen.clicked.connect(sharpen_selected)
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

    def start_photo_loader(
        self, files, *, pose=None, folder=None, edit_targets=None,
        image_overrides=None, operation=None, on_finished=None
    ):
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
        elif edit_targets:
            self.status.setText("Re-aligning edited faces...")
        else:
            self.status.setText(f"Finding faces in {len(files)} selected photo(s)...")
        self.folder_button.setEnabled(False)
        self.save.setEnabled(False)
        self.loaded_photo_count = 0
        self.pending_pose = pose
        self.edit_targets = edit_targets
        self.edit_operation = operation
        self.on_photo_load_finished = on_finished
        self.loader = PosePhotoLoader(
            self.main_window, files_list=files, parent=self,
            max_faces_per_image=0 if folder else 1, align_preview=True, image_overrides=image_overrides,
        )
        self.pending_landmarks = None
        self.loader.mesh_ready.connect(self.mesh_loaded)
        self.loader.thumbnail_ready.connect(self.photo_loaded)
        self.loader.finished.connect(self.finish_loading, QtCore.Qt.ConnectionType.QueuedConnection)
        self.loader.start()

    @QtCore.Slot(object)
    def mesh_loaded(self, points):
        self.pending_landmarks = points

    @QtCore.Slot(str, np.ndarray, object, QtGui.QImage, str)
    def photo_loaded(self, path, _crop, embeddings, image, _face_id):
        photo = PosePhoto(
            Path(path).name, aligned_rgb(image), embeddings, landmarks=self.pending_landmarks
        )
        self.pending_landmarks = None
        if self.pending_pose is None:
            self.gallery_photos.append(photo)
            item = QtWidgets.QListWidgetItem(self.face_icon(photo), "")
            item.setToolTip(photo.name)
            item.setData(QtCore.Qt.ItemDataRole.AccessibleTextRole, photo.name)
            self.gallery.addItem(item)
        elif self.edit_targets:
            original = self.edit_targets[path]
            slot = self.photos.get(self.pending_pose, [])
            if original in slot:
                photo.name = original.name
                photo.excluded = original.excluded
                slot[slot.index(original)] = photo
                self.loaded_photo_count += 1
        else:
            self.loaded_photo_count += 1
            self.add_pose_photos(self.pending_pose, [photo])
            self.load_refresh.start()

    @QtCore.Slot()
    def finish_loading(self):
        if self.loader is None:
            return
        pose = self.pending_pose
        self.load_refresh.stop()
        self.loader.wait()
        self.loader.deleteLater()
        self.loader = None
        self.folder_button.setEnabled(True)
        if self.edit_targets:
            attempted = len(self.edit_targets)
            self.edit_targets = None
            callback = self.on_photo_load_finished
            self.on_photo_load_finished = None
            self.refresh()
            count = self.loaded_photo_count
            message = f"{self.edit_operation} {count} of {attempted} selected photo(s) in {pose}."
            if count < attempted:
                message += " Photos that could not be aligned were kept."
            self.status.setText(message)
            if callback:
                callback()
            return
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
            real = [photo.rgb for photo in self.photos.get(source, []) if not photo.origin and not photo.excluded]
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
        real = [photo.rgb for slot in self.photos.values() for photo in slot if not photo.origin and not photo.excluded]
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
                if not photo.origin and not photo.excluded
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
        photos = list(dict.fromkeys(
            photo for slot in self.photos.values() for photo in slot if not photo.excluded
        ))
        real = [photo for photo in photos if not photo.origin]
        front = [
            photo for photo in self.photos.get("Front", []) if not photo.origin and not photo.excluded
        ]
        mode = self.preview_mode.currentText()
        self.preview.mesh = None
        self.preview.wireframe = mode == "Wireframe"
        self.preview.render_key = None
        if real and mode != "Mean / Median composite":
            key = tuple(real), tuple(front)
            if key != self.texture_cache_key:
                self.texture_cache = fuse_face_texture(real, front)
                self.texture_cache_key = key
            texture, points, triangles = self.texture_cache
            self.preview.image = image_from_rgb(texture)
            if mode in ("Textured 3D", "Wireframe") and points is not None:
                self.preview.mesh = texture, points, triangles
                self.preview_hint.setText("Drag to rotate the facial surface.")
            elif mode in ("Textured 3D", "Wireframe"):
                self.preview_hint.setText("3D landmarks unavailable. Showing the best available crop.")
            else:
                self.preview_hint.setText("Sharp reference anchored with aligned, color-matched detail.")
        else:
            self.preview.image = (
                image_from_rgb(np.rint(reduce_samples([photo.rgb for photo in photos], method)))
                if photos else QtGui.QImage()
            )
            self.preview_hint.setText("Drag to tilt the composite.")
        self.preview.update()
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
