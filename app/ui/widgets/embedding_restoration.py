"""Preview VisoMaster's CodeFormer restoration on an embedding reference."""

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


def restore_crop(main_window, rgb):
    """Run the existing restorer, aligned to its FFHQ reference coordinates."""
    import torch

    from app.processors.utils.faceutil import get_arcface_template

    processor = main_window.models_processor
    worker = main_window.function_worker
    if not processor.load_model("CodeFormer"):
        raise RuntimeError("CodeFormer could not be loaded.")
    pixels = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
    image = torch.from_numpy(pixels).permute(2, 0, 1).to(processor.device)
    restored = worker.apply_facerestorer(
        image, "Reference", "CodeFormer", 100.0, 0.95, 0.5,
        get_arcface_template(512, "arcface112")[0],
    )
    if not torch.isfinite(restored).all().item():
        raise RuntimeError("CodeFormer returned an invalid image.")
    pixels = restored.permute(1, 2, 0).round().clamp(0, 255).byte().cpu().numpy()
    return cv2.resize(pixels, (224, 224), interpolation=cv2.INTER_AREA)


def blend_restoration(original, restored, strength):
    # Fade out at the crop boundaries, where the inverse warp can leave gaps.
    yy, xx = np.ogrid[:224, :224]
    radius = np.sqrt(((xx - 112) / 103) ** 2 + ((yy - 121) / 105) ** 2)
    weight = np.clip((1 - radius) / 0.22, 0, 1) * (strength / 100)
    return np.rint(original + (restored.astype(np.float32) - original) * weight[..., None]).astype(np.uint8)


class RestorationWorker(QtCore.QThread):
    restored = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, main_window, rgb, parent):
        super().__init__(parent)
        self.main_window = main_window
        self.rgb = rgb

    def run(self):
        try:
            self.restored.emit(restore_crop(self.main_window, self.rgb))
        except Exception as error:
            self.failed.emit(str(error))


class RestorationPreview(QtWidgets.QDialog):
    def __init__(self, parent, main_window, original):
        super().__init__(parent)
        self.setWindowTitle("Restore Reference with CodeFormer")
        self.original = original
        self.restored = None
        self.result = None
        self.worker = RestorationWorker(main_window, original, self)
        self.worker.restored.connect(self.set_restored)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.restoration_finished)

        layout = QtWidgets.QVBoxLayout(self)
        images = QtWidgets.QHBoxLayout()
        for label, rgb in (("Original", original), ("Restored blend", None)):
            column = QtWidgets.QVBoxLayout()
            column.addWidget(QtWidgets.QLabel(label))
            preview = QtWidgets.QLabel()
            preview.setFixedSize(224, 224)
            if rgb is None:
                self.restored_preview = preview
            else:
                self.show_image(preview, rgb)
            column.addWidget(preview)
            images.addLayout(column)
        layout.addLayout(images)

        row = QtWidgets.QHBoxLayout()
        self.strength_label = QtWidgets.QLabel()
        row.addWidget(self.strength_label)
        self.strength = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.strength.setRange(0, 100)
        self.strength.setValue(35)
        self.strength.setEnabled(False)
        self.strength.setAccessibleName("Restoration blend")
        self.strength.valueChanged.connect(self.refresh_preview)
        row.addWidget(self.strength, 1)
        layout.addLayout(row)
        self.status = QtWidgets.QLabel("Generating a high-fidelity preview...")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.addButton("Use restored crop", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.apply_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh_preview()

    @staticmethod
    def show_image(widget, rgb):
        pixels = np.ascontiguousarray(rgb)
        image = QtGui.QImage(
            pixels.data, 224, 224, pixels.strides[0], QtGui.QImage.Format.Format_RGB888
        )
        widget.setPixmap(QtGui.QPixmap.fromImage(image))

    def exec(self):
        self.worker.start()
        return super().exec()

    def done(self, result):
        if self.worker.isRunning():
            self.status.setText("CodeFormer is still preparing the preview.")
            return
        super().done(result)

    @QtCore.Slot(object)
    def set_restored(self, rgb):
        self.restored = rgb
        self.status.setText("Compare both crops before using the restoration.")
        self.refresh_preview()

    @QtCore.Slot(str)
    def show_error(self, error):
        self.status.setText(f"Could not restore this crop. {error}")

    @QtCore.Slot()
    def restoration_finished(self):
        self.strength.setEnabled(self.restored is not None)
        self.refresh_preview()

    def refresh_preview(self):
        strength = self.strength.value()
        self.strength_label.setText(f"Restoration blend {strength}%")
        if self.restored is not None:
            self.result = blend_restoration(self.original, self.restored, strength)
            self.show_image(self.restored_preview, self.result)
        self.apply_button.setEnabled(
            self.result is not None and strength > 0 and not self.worker.isRunning()
        )
