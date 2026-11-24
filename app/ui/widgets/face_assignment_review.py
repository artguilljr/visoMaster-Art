"""Review and edit the assignments on the current target face cards."""

from pathlib import Path

from PySide6 import QtCore, QtWidgets


def open_review(main_window):
    from app.ui.widgets.actions import card_actions, common_actions, video_control_actions

    if video_control_actions.block_if_issue_scan_active(main_window, "review faces"):
        return
    video = main_window.video_processor
    if (
        main_window.buttonMediaRecord.isChecked()
        or video.is_processing_segments
        or getattr(main_window, "is_batch_processing", False)
    ):
        QtWidgets.QMessageBox.information(
            main_window, "Face Assignment Review", "Finish rendering before reviewing faces."
        )
        return
    if not main_window.target_faces:
        QtWidgets.QMessageBox.information(
            main_window, "Face Assignment Review",
            "Open target media and use Find Faces to add people to review.",
        )
        return

    dialog = FaceAssignmentReview(main_window)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return
    if not dialog.apply_changes():
        return

    selected = main_window.target_faces.get(main_window.selected_target_face_id)
    if selected is not None:
        card_actions.uncheck_all_input_faces(main_window)
        card_actions.uncheck_all_merged_embeddings(main_window)
        for face_id in selected.assigned_input_faces:
            if face_id in main_window.input_faces:
                main_window.input_faces[face_id].setChecked(True)
        for embedding_id in selected.assigned_merged_embeddings:
            if embedding_id in main_window.merged_embeddings:
                main_window.merged_embeddings[embedding_id].setChecked(True)
    common_actions.refresh_frame(main_window)


class FaceAssignmentReview(QtWidgets.QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle("Face Assignment Review")
        self.resize(760, 420)
        self.main_window = main_window
        self.assignments = []

        layout = QtWidgets.QVBoxLayout(self)
        global_assignment = any(
            main_window.control.get(key) for key in ("KeepInputToggle", "AutoSwapToggle")
        )
        instructions = (
            "Turn off Keep Input and Auto Swap to edit individual assignments."
            if global_assignment
            else "Review detected faces and choose a replacement or leave a face untouched."
        )
        message = QtWidgets.QLabel(instructions)
        message.setWordWrap(True)
        layout.addWidget(message)

        sources = {
            ("input", key): (Path(card.media_path).name or f"Input {index}", card)
            for index, (key, card) in enumerate(main_window.input_faces.items(), 1)
        }
        sources.update({
            ("embedding", key): (card.embedding_name, card)
            for key, card in main_window.merged_embeddings.items()
        })

        table = QtWidgets.QTableWidget(len(main_window.target_faces), 4)
        table.setHorizontalHeaderLabels(
            ["Target face", "Observed frames", "Current replacement", "New replacement"]
        )
        table.setToolTip("Observed frames come from frames checked with Find Faces.")
        table.setIconSize(QtCore.QSize(56, 56))
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().hide()
        table.verticalHeader().setDefaultSectionSize(64)
        header = table.horizontalHeader()
        for column in (2, 3):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)

        for row, target in enumerate(main_window.target_faces.values()):
            table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(target.icon(), target.get_display_label())
            )
            observed = getattr(target, "observed_frame_range", None)
            frames = "—"
            if observed is not None:
                first, last = (frame + 1 for frame in observed)
                frames = str(first) if first == last else f"{first}–{last}"
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(frames))

            assigned = [("input", key) for key in target.assigned_input_faces]
            assigned += [("embedding", key) for key in target.assigned_merged_embeddings]
            names = ", ".join(sources.get(key, ("Unavailable source", None))[0] for key in assigned)
            current = QtWidgets.QTableWidgetItem(names or "Untouched")
            current.setToolTip(names)
            table.setItem(row, 2, current)

            choice = QtWidgets.QComboBox()
            choice.setAccessibleName(f"New replacement for {target.get_display_label()}")
            choice.addItem("Keep current", None)
            choice.addItem("Leave untouched", ("clear", None))
            for (kind, key), (label, card) in sources.items():
                prefix = "Photo" if kind == "input" else "Embedding"
                choice.addItem(card.icon(), f"{prefix} · {label}", (kind, key))
            choice.setEnabled(not global_assignment)
            table.setCellWidget(row, 3, choice)
            self.assignments.append((target.face_id, choice))

        standard = QtWidgets.QDialogButtonBox.StandardButton
        buttons = QtWidgets.QDialogButtonBox(
            standard.Close if global_assignment else standard.Ok | standard.Cancel
        )
        if not global_assignment:
            buttons.button(standard.Ok).setText("Apply changes")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply_changes(self):
        changed = False
        for face_id, choice in self.assignments:
            selection = choice.currentData()
            target = self.main_window.target_faces.get(face_id)
            if selection is None or target is None:
                continue
            kind, source_id = selection
            inputs = {}
            embeddings = {}
            if kind == "input":
                source = self.main_window.input_faces.get(source_id)
                if source is None:
                    continue
                inputs[source_id] = source.embedding_store
            elif kind == "embedding":
                source = self.main_window.merged_embeddings.get(source_id)
                if source is None:
                    continue
                embeddings[source_id] = source.embedding_store
            if (
                target.assigned_input_faces.keys() == inputs.keys()
                and target.assigned_merged_embeddings.keys() == embeddings.keys()
            ):
                continue
            target.assigned_input_faces = inputs
            target.assigned_merged_embeddings = embeddings
            target.calculate_assigned_input_embedding()
            changed = True
        return changed
