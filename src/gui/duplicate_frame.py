"""
duplicate_frame.py

This module implements the Duplicate Frame for the Entity Validation System.
It provides a UI for detecting and managing duplicate entities.
"""

import os
import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHeaderView,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QSplitter,
    QFrame,
    QProgressBar,
    QApplication,
    QCheckBox,
    QGroupBox,
    QStyleFactory,
    QTabWidget,
    QRadioButton,
    QButtonGroup,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QIcon

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import validation modules
from validation.duplicate_detection import DuplicateDetector
from utils.fuzzy_matching import EntityMatcher


class DuplicateWorker(QThread):
    """Worker thread for running duplicate detection in the background."""

    # Signals
    progress_updated = pyqtSignal(int)
    internal_duplicates_completed = pyqtSignal(object, dict)
    external_duplicates_completed = pyqtSignal(object, dict)
    duplicate_detection_failed = pyqtSignal(str)

    def __init__(self, df, auth_list_path, fuzzy_threshold=None, entity_columns=None):
        """Initialize the duplicate worker.

        Args:
            df (pandas.DataFrame): DataFrame with entity data.
            auth_list_path (str): Path to the authoritative entity list.
            fuzzy_threshold (float, optional): Threshold for fuzzy matching.
                Defaults to None (use default threshold).
            entity_columns (list, optional): List of columns containing entity names.
                Defaults to None (use default entity columns).
        """
        super().__init__()
        self.df = df
        self.auth_list_path = auth_list_path
        self.fuzzy_threshold = fuzzy_threshold
        self.entity_columns = entity_columns
        self.detector = None

    def run(self):
        """Run duplicate detection in a separate thread."""
        try:
            # Initialize detector
            self.detector = DuplicateDetector(
                auth_list_path=self.auth_list_path, fuzzy_threshold=self.fuzzy_threshold
            )

            # Check if dataframe is valid
            if self.df is None or self.df.empty:
                self.duplicate_detection_failed.emit(
                    "Input DataFrame is empty or invalid."
                )
                return

            # Step 1: Detect internal duplicates
            self.progress_updated.emit(25)
            # Process smaller batches to update progress more frequently
            batch_size = 100
            total_rows = len(self.df)

            # Handle internal duplicates in smaller batches for progress updates
            internal_dup_df, duplicate_groups = (
                self.detector.detect_internal_duplicates(self.df, self.entity_columns)
            )

            # Emit results for internal duplicates
            self.internal_duplicates_completed.emit(internal_dup_df, duplicate_groups)

            # Step 2: Detect duplicates against authoritative list
            self.progress_updated.emit(50)

            # Process external duplicates in batches
            external_dup_df, match_info = self.detector.detect_duplicates(
                internal_dup_df,
                self.entity_columns,
                report_progress=lambda p: self.progress_updated.emit(50 + int(p * 0.5)),
            )

            # Emit results for external duplicates
            self.progress_updated.emit(100)
            self.external_duplicates_completed.emit(external_dup_df, match_info)

        except Exception as e:
            self.duplicate_detection_failed.emit(
                f"Duplicate detection failed: {str(e)}"
            )


class DuplicateFrame(QWidget):
    """Frame for detecting and managing duplicate entities."""

    # Signals
    duplicate_status_changed = pyqtSignal(bool)  # True if duplicate detection complete
    continue_to_reports = pyqtSignal()  # Signal to continue to reports

    def __init__(self, parent=None):
        """Initialize the duplicate frame."""
        super().__init__(parent)
        self.input_df = None
        self.internal_dup_df = None
        self.external_dup_df = None
        self.duplicate_groups = {}
        self.match_info = {}
        self.auth_list_path = None
        self.fuzzy_threshold = 85  # Default threshold
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Status group
        status_group = QGroupBox("Duplicate Detection Status")
        status_layout = QVBoxLayout(status_group)

        # Status label
        self.status_label = QLabel("No duplicate detection has been performed yet.")
        status_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        # Summary statistics
        self.stats_label = QLabel(
            "Summary statistics will appear here after duplicate detection."
        )
        status_layout.addWidget(self.stats_label)

        main_layout.addWidget(status_group)

        # Tabs for internal and external duplicates
        self.duplicate_tabs = QTabWidget()

        # Internal duplicates tab
        self.internal_tab = QWidget()
        internal_layout = QVBoxLayout(self.internal_tab)

        # Table for internal duplicates
        self.internal_table = QTableWidget()
        self.internal_table.setColumnCount(5)
        self.internal_table.setHorizontalHeaderLabels(
            ["Entity", "Duplicate Group", "Match", "Score", "Action"]
        )
        self.internal_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.internal_table.setAlternatingRowColors(True)
        internal_layout.addWidget(self.internal_table)

        # Action buttons for internal duplicates
        internal_action_layout = QHBoxLayout()

        self.resolve_selected_btn = QPushButton("Resolve Selected Group")
        self.resolve_selected_btn.setEnabled(False)
        self.resolve_selected_btn.clicked.connect(self.resolve_selected_group)
        internal_action_layout.addWidget(self.resolve_selected_btn)

        self.resolve_all_btn = QPushButton("Resolve All Groups")
        self.resolve_all_btn.setEnabled(False)
        self.resolve_all_btn.clicked.connect(self.resolve_all_groups)
        internal_action_layout.addWidget(self.resolve_all_btn)

        internal_layout.addLayout(internal_action_layout)

        # External duplicates tab
        self.external_tab = QWidget()
        external_layout = QVBoxLayout(self.external_tab)

        # Table for external duplicates
        self.external_table = QTableWidget()
        self.external_table.setColumnCount(6)
        self.external_table.setHorizontalHeaderLabels(
            ["Entity", "Auth Match", "Score", "Exact Match", "Action", "Status"]
        )
        self.external_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.external_table.setAlternatingRowColors(True)
        external_layout.addWidget(self.external_table)

        # Action buttons for external duplicates
        external_action_layout = QHBoxLayout()

        self.accept_all_matches_btn = QPushButton("Accept All Matches")
        self.accept_all_matches_btn.setEnabled(False)
        self.accept_all_matches_btn.clicked.connect(self.accept_all_matches)
        external_action_layout.addWidget(self.accept_all_matches_btn)

        self.reject_all_matches_btn = QPushButton("Reject All Matches")
        self.reject_all_matches_btn.setEnabled(False)
        self.reject_all_matches_btn.clicked.connect(self.reject_all_matches)
        external_action_layout.addWidget(self.reject_all_matches_btn)

        external_layout.addLayout(external_action_layout)

        # Add tabs to the tab widget
        self.duplicate_tabs.addTab(self.internal_tab, "Internal Duplicates")
        self.duplicate_tabs.addTab(self.external_tab, "External Duplicates")

        main_layout.addWidget(self.duplicate_tabs)

        # Buttons layout at bottom
        buttons_layout = QHBoxLayout()

        # Apply changes button
        self.apply_changes_btn = QPushButton("Apply Changes")
        self.apply_changes_btn.setEnabled(False)
        self.apply_changes_btn.clicked.connect(self.apply_changes)
        buttons_layout.addWidget(self.apply_changes_btn)

        # Continue to Reports button
        self.continue_to_reports_btn = QPushButton("Continue to Reports")
        self.continue_to_reports_btn.setEnabled(False)
        self.continue_to_reports_btn.clicked.connect(self.continue_to_reports.emit)
        buttons_layout.addWidget(self.continue_to_reports_btn)

        main_layout.addLayout(buttons_layout)

    def set_data(self, df, auth_list_path, fuzzy_threshold=None, entity_columns=None):
        """Set data for duplicate detection.

        Args:
            df (pandas.DataFrame): DataFrame to check for duplicates.
            auth_list_path (str): Path to the authoritative entity list.
            fuzzy_threshold (float, optional): Threshold for fuzzy matching.
                Defaults to None (use default threshold).
            entity_columns (list, optional): List of columns containing entity names.
                Defaults to None (use default entity columns).
        """
        if df is None or df.empty:
            QMessageBox.warning(self, "Invalid Data", "DataFrame is empty or invalid.")
            return

        if not os.path.exists(auth_list_path):
            QMessageBox.warning(
                self,
                "Invalid Path",
                f"Authoritative list file not found: {auth_list_path}",
            )
            return

        # Store parameters
        self.input_df = df
        self.auth_list_path = auth_list_path
        if fuzzy_threshold is not None:
            self.fuzzy_threshold = fuzzy_threshold

        # Reset UI
        self.reset_ui()

        # Start duplicate detection in worker thread
        self.worker = DuplicateWorker(
            df, auth_list_path, self.fuzzy_threshold, entity_columns
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.internal_duplicates_completed.connect(
            self.display_internal_duplicates
        )
        self.worker.external_duplicates_completed.connect(
            self.display_external_duplicates
        )
        self.worker.duplicate_detection_failed.connect(self.handle_detection_error)

        # Update status
        self.status_label.setText("Detecting duplicates...")
        self.progress_bar.setValue(0)

        # Start worker thread
        self.worker.start()

    @pyqtSlot(int)
    def update_progress(self, value):
        """Update progress bar.

        Args:
            value (int): Progress percentage (0-100).
        """
        self.progress_bar.setValue(value)
        QApplication.processEvents()  # Allow UI to update

    @pyqtSlot(object, dict)
    def display_internal_duplicates(self, internal_dup_df, duplicate_groups):
        """Display internal duplicate detection results.

        Args:
            internal_dup_df (pandas.DataFrame): DataFrame with internal duplicate info.
            duplicate_groups (dict): Dictionary of duplicate groups.
        """
        self.internal_dup_df = internal_dup_df
        self.duplicate_groups = duplicate_groups

        # Find entities that are part of duplicate groups
        duplicate_entities = internal_dup_df[
            internal_dup_df["is_internal_duplicate"]
        ].copy()

        # Update table
        self.internal_table.setRowCount(len(duplicate_entities))

        row = 0
        for idx, entity_row in duplicate_entities.iterrows():
            # Only process rows with actual duplicate groups
            if pd.isna(entity_row["internal_duplicate_group"]):
                continue

            # Entity name
            entity_name = None
            for col in ["name", "facility", "owner", "operator"]:
                if col in entity_row and pd.notna(entity_row[col]):
                    entity_name = entity_row[col]
                    break

            if not entity_name:
                continue

            entity_item = QTableWidgetItem(str(entity_name))
            entity_item.setFlags(entity_item.flags() & ~Qt.ItemIsEditable)
            self.internal_table.setItem(row, 0, entity_item)

            # Duplicate group
            group_id = entity_row["internal_duplicate_group"]
            group_item = QTableWidgetItem(str(group_id))
            group_item.setFlags(group_item.flags() & ~Qt.ItemIsEditable)
            self.internal_table.setItem(row, 1, group_item)

            # Match
            match = entity_row["internal_duplicate_match"]
            match_item = QTableWidgetItem(str(match) if pd.notna(match) else "")
            match_item.setFlags(match_item.flags() & ~Qt.ItemIsEditable)
            self.internal_table.setItem(row, 2, match_item)

            # Score
            score = entity_row["internal_duplicate_score"]
            score_item = QTableWidgetItem(f"{score:.3f}" if pd.notna(score) else "")
            score_item.setFlags(score_item.flags() & ~Qt.ItemIsEditable)
            self.internal_table.setItem(row, 3, score_item)

            # Action combo box
            action_widget = QComboBox()
            action_widget.addItems(["Choose Action", "Keep Best", "Merge", "Custom"])
            action_widget.setCurrentIndex(0)
            action_widget.setProperty("row", row)
            action_widget.setProperty("group_id", group_id)
            action_widget.setProperty("entity_name", entity_name)
            action_widget.currentIndexChanged.connect(self.on_internal_action_changed)
            self.internal_table.setCellWidget(row, 4, action_widget)

            row += 1

            # Update UI every 100 rows to keep it responsive
            if row % 100 == 0:
                QApplication.processEvents()

        # Update table row count to actual number of rows added
        if row < self.internal_table.rowCount():
            self.internal_table.setRowCount(row)

        # Enable action buttons if there are duplicate groups
        has_duplicates = len(duplicate_groups) > 0
        self.resolve_selected_btn.setEnabled(has_duplicates)
        self.resolve_all_btn.setEnabled(has_duplicates)

        # Update statistics
        total_entities = len(internal_dup_df)
        duplicate_count = len(duplicate_entities)
        duplicate_groups_count = len(duplicate_groups)

        # Update stats label with internal duplicate info
        self.stats_label.setText(
            f"Total entities: {total_entities}, "
            f"Internal duplicates: {duplicate_count}, "
            f"Duplicate groups: {duplicate_groups_count}"
        )

    @pyqtSlot(object, dict)
    def display_external_duplicates(self, external_dup_df, match_info):
        """Display external duplicate detection results.

        Args:
            external_dup_df (pandas.DataFrame): DataFrame with external duplicate info.
            match_info (dict): Dictionary of match information.
        """
        self.external_dup_df = external_dup_df
        self.match_info = match_info

        # Find entities with potential matches
        matched_entities = external_dup_df[
            external_dup_df["duplicate_match"].notna()
        ].copy()

        # Update table
        self.external_table.setRowCount(len(matched_entities))

        row = 0
        for idx, entity_row in matched_entities.iterrows():
            # Entity name
            entity_name = None
            for col in ["name", "facility", "owner", "operator"]:
                if col in entity_row and pd.notna(entity_row[col]):
                    entity_name = entity_row[col]
                    break

            if not entity_name:
                continue

            entity_item = QTableWidgetItem(str(entity_name))
            entity_item.setFlags(entity_item.flags() & ~Qt.ItemIsEditable)
            self.external_table.setItem(row, 0, entity_item)

            # Auth match
            match = entity_row["duplicate_match"]
            match_item = QTableWidgetItem(str(match) if pd.notna(match) else "")
            match_item.setFlags(match_item.flags() & ~Qt.ItemIsEditable)
            self.external_table.setItem(row, 1, match_item)

            # Score
            score = entity_row["duplicate_score"]
            score_item = QTableWidgetItem(f"{score:.3f}" if pd.notna(score) else "")
            score_item.setFlags(score_item.flags() & ~Qt.ItemIsEditable)
            self.external_table.setItem(row, 2, score_item)

            # Exact match
            is_exact = entity_row["is_exact_match"]
            exact_item = QTableWidgetItem("Yes" if is_exact else "No")
            exact_item.setFlags(exact_item.flags() & ~Qt.ItemIsEditable)
            self.external_table.setItem(row, 3, exact_item)

            # Action combo box (disabled for exact matches)
            action_widget = QComboBox()
            action_widget.addItems(
                ["Choose Action", "Accept Match", "Reject Match", "New Entity"]
            )
            action_widget.setCurrentIndex(0)
            action_widget.setProperty("row", row)
            action_widget.setProperty("entity_name", entity_name)

            # Disable combo box for exact matches
            if is_exact:
                action_widget.setCurrentIndex(1)  # Accept Match
                action_widget.setEnabled(False)

            action_widget.currentIndexChanged.connect(self.on_external_action_changed)
            self.external_table.setCellWidget(row, 4, action_widget)

            # Status
            status_item = QTableWidgetItem("Exact Match" if is_exact else "Pending")
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            self.external_table.setItem(row, 5, status_item)

            # Set row color based on status
            if is_exact:
                self.set_row_color(
                    self.external_table, row, QColor(200, 255, 200)
                )  # Light green

            row += 1

            # Update UI every 100 rows to keep it responsive
            if row % 100 == 0:
                QApplication.processEvents()

        # Update table row count to actual number of rows added
        if row < self.external_table.rowCount():
            self.external_table.setRowCount(row)

        # Enable action buttons if there are matches
        has_matches = len(matched_entities) > 0
        self.accept_all_matches_btn.setEnabled(has_matches)
        self.reject_all_matches_btn.setEnabled(has_matches)

        # Update statistics
        total_entities = len(external_dup_df)
        matched_count = len(matched_entities)
        exact_match_count = matched_entities["is_exact_match"].sum()

        # Update final status
        self.status_label.setText("Duplicate detection complete.")
        self.progress_bar.setValue(100)

        # Append external duplicate info to stats label
        self.stats_label.setText(
            self.stats_label.text() + f", External matches: {matched_count}, "
            f"Exact matches: {exact_match_count}"
        )

        # Enable apply changes button and continue to reports button
        self.apply_changes_btn.setEnabled(True)
        self.continue_to_reports_btn.setEnabled(True)

        # Signal that duplicate detection is complete
        self.duplicate_status_changed.emit(True)

    @pyqtSlot(str)
    def handle_detection_error(self, error_message):
        """Handle duplicate detection errors.

        Args:
            error_message (str): Error message to display.
        """
        self.status_label.setText(f"Error: {error_message}")
        self.progress_bar.setValue(0)

        QMessageBox.critical(self, "Duplicate Detection Error", error_message)

        # Signal that duplicate detection failed
        self.duplicate_status_changed.emit(False)

    def on_internal_action_changed(self, index):
        """Handle internal duplicates action combo box changes.

        Args:
            index (int): Selected index in the combo box.
        """
        combo_box = self.sender()
        row = combo_box.property("row")
        group_id = combo_box.property("group_id")
        entity_name = combo_box.property("entity_name")

        if index == 1:  # Keep Best
            # Find the best entity in the group
            if group_id in self.duplicate_groups:
                for r in range(self.internal_table.rowCount()):
                    group_item = self.internal_table.item(r, 1)
                    if group_item and group_item.text() == group_id:
                        status_item = QTableWidgetItem("Keep Best")
                        if (
                            self.internal_table.columnCount() > 5
                        ):  # If status column exists
                            self.internal_table.setItem(r, 5, status_item)
                        self.set_row_color(
                            self.internal_table, r, QColor(200, 255, 200)
                        )  # Light green

        elif index == 2:  # Merge
            # Mark all entities in the group for merging
            if group_id in self.duplicate_groups:
                for r in range(self.internal_table.rowCount()):
                    group_item = self.internal_table.item(r, 1)
                    if group_item and group_item.text() == group_id:
                        status_item = QTableWidgetItem("Merge")
                        if (
                            self.internal_table.columnCount() > 5
                        ):  # If status column exists
                            self.internal_table.setItem(r, 5, status_item)
                        self.set_row_color(
                            self.internal_table, r, QColor(200, 200, 255)
                        )  # Light blue

        elif index == 3:  # Custom
            # Create a custom entry
            if group_id in self.duplicate_groups:
                custom_value, ok = QLineEdit.getText(
                    self,
                    "Custom Entity Name",
                    "Enter custom entity name for this group:",
                    QLineEdit.Normal,
                    entity_name,
                )

                if ok and custom_value:
                    for r in range(self.internal_table.rowCount()):
                        group_item = self.internal_table.item(r, 1)
                        if group_item and group_item.text() == group_id:
                            status_item = QTableWidgetItem(f"Custom: {custom_value}")
                            if (
                                self.internal_table.columnCount() > 5
                            ):  # If status column exists
                                self.internal_table.setItem(r, 5, status_item)
                            self.set_row_color(
                                self.internal_table, r, QColor(255, 200, 255)
                            )  # Light purple

    def on_external_action_changed(self, index):
        """Handle external duplicates action combo box changes.

        Args:
            index (int): Selected index in the combo box.
        """
        combo_box = self.sender()
        row = combo_box.property("row")
        entity_name = combo_box.property("entity_name")

        if row >= self.external_table.rowCount():
            return

        if index == 1:  # Accept Match
            # Update status
            status_item = QTableWidgetItem("Accepted")
            self.external_table.setItem(row, 5, status_item)

            # Update row color
            self.set_row_color(
                self.external_table, row, QColor(200, 255, 200)
            )  # Light green

        elif index == 2:  # Reject Match
            # Update status
            status_item = QTableWidgetItem("Rejected")
            self.external_table.setItem(row, 5, status_item)

            # Update row color
            self.set_row_color(
                self.external_table, row, QColor(255, 200, 200)
            )  # Light red

        elif index == 3:  # New Entity
            # Update status
            status_item = QTableWidgetItem("New Entity")
            self.external_table.setItem(row, 5, status_item)

            # Update row color
            self.set_row_color(
                self.external_table, row, QColor(255, 255, 200)
            )  # Light yellow

    def set_row_color(self, table, row, color):
        """Set the background color for a table row.

        Args:
            table (QTableWidget): The table widget.
            row (int): Row index.
            color (QColor): Background color.
        """
        for col in range(table.columnCount()):
            item = table.item(row, col)
            if item:
                item.setBackground(color)

    def resolve_selected_group(self):
        """Resolve the selected duplicate group."""
        # Get the selected row
        selected_items = self.internal_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a row first.")
            return

        # Get the group ID from the selected row
        row = selected_items[0].row()
        group_item = self.internal_table.item(row, 1)
        if not group_item:
            return

        group_id = group_item.text()

        # Get the action for this group
        action_widget = self.internal_table.cellWidget(row, 4)
        if not action_widget:
            return

        # Simulate selecting "Keep Best" if no action is selected
        if action_widget.currentIndex() == 0:
            action_widget.setCurrentIndex(1)  # Keep Best

    def resolve_all_groups(self):
        """Resolve all duplicate groups."""
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Resolve All Groups",
            "This will resolve all duplicate groups using the 'Keep Best' option. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        # Apply "Keep Best" action to all groups
        for row in range(self.internal_table.rowCount()):
            action_widget = self.internal_table.cellWidget(row, 4)
            if action_widget and action_widget.currentIndex() == 0:
                action_widget.setCurrentIndex(1)  # Keep Best

    def accept_all_matches(self):
        """Accept all duplicate matches."""
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Accept All Matches",
            "This will accept all external duplicate matches. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        # Apply "Accept Match" action to all non-exact matches
        for row in range(self.external_table.rowCount()):
            exact_item = self.external_table.item(row, 3)
            action_widget = self.external_table.cellWidget(row, 4)

            # Skip exact matches and disabled action widgets
            if (
                (not exact_item or exact_item.text() != "Yes")
                and action_widget
                and action_widget.isEnabled()
            ):
                action_widget.setCurrentIndex(1)  # Accept Match

    def reject_all_matches(self):
        """Reject all duplicate matches."""
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Reject All Matches",
            "This will reject all external duplicate matches. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        # Apply "Reject Match" action to all non-exact matches
        for row in range(self.external_table.rowCount()):
            exact_item = self.external_table.item(row, 3)
            action_widget = self.external_table.cellWidget(row, 4)

            # Skip exact matches and disabled action widgets
            if (
                (not exact_item or exact_item.text() != "Yes")
                and action_widget
                and action_widget.isEnabled()
            ):
                action_widget.setCurrentIndex(2)  # Reject Match

    def apply_changes(self):
        """Apply duplicate resolution changes."""
        if not self.internal_dup_df is not None or not self.external_dup_df is not None:
            QMessageBox.warning(
                self,
                "Cannot Apply Changes",
                "No duplicate detection results available.",
            )
            return

        # Process internal duplicate decisions
        internal_decisions = {}
        for row in range(self.internal_table.rowCount()):
            group_item = self.internal_table.item(row, 1)
            action_widget = self.internal_table.cellWidget(row, 4)

            if not group_item or not action_widget:
                continue

            group_id = group_item.text()
            action_idx = action_widget.currentIndex()

            if action_idx > 0 and group_id not in internal_decisions:
                if action_idx == 1:  # Keep Best
                    internal_decisions[group_id] = {
                        "action": "keep",
                        "keep_entity": None,  # Will be determined by the system
                    }
                elif action_idx == 2:  # Merge
                    internal_decisions[group_id] = {"action": "merge"}
                elif action_idx == 3:  # Custom
                    status_item = None
                    if self.internal_table.columnCount() > 5:
                        status_item = self.internal_table.item(row, 5)

                    if status_item and status_item.text().startswith("Custom: "):
                        custom_value = status_item.text()[
                            8:
                        ]  # Remove "Custom: " prefix
                        internal_decisions[group_id] = {
                            "action": "custom",
                            "custom_value": custom_value,
                        }

        # Process external duplicate decisions
        external_decisions = {}
        for row in range(self.external_table.rowCount()):
            entity_item = self.external_table.item(row, 0)
            match_item = self.external_table.item(row, 1)
            action_widget = self.external_table.cellWidget(row, 4)

            if not entity_item or not match_item or not action_widget:
                continue

            entity_name = entity_item.text()
            match_name = match_item.text()
            action_idx = action_widget.currentIndex()

            if action_idx > 0:
                if action_idx == 1:  # Accept Match
                    external_decisions[entity_name] = {
                        "action": "accept",
                        "selected_match": match_name,
                    }
                elif action_idx == 2:  # Reject Match
                    external_decisions[entity_name] = {"action": "reject"}
                elif action_idx == 3:  # New Entity
                    external_decisions[entity_name] = {"action": "new"}

        # Create DuplicateDetector instance
        detector = DuplicateDetector(self.auth_list_path)

        # Apply internal duplicate decisions
        if internal_decisions:
            result_df = detector.handle_internal_duplicates(
                self.internal_dup_df, self.duplicate_groups, internal_decisions
            )
        else:
            result_df = self.internal_dup_df

        # Apply external duplicate decisions
        if external_decisions:
            result_df, new_entities = detector.process_user_decisions(
                result_df, self.match_info, external_decisions
            )

            # Update authoritative list with new entities
            if new_entities:
                detector.update_authoritative_list(new_entities)

        # Show confirmation
        QMessageBox.information(
            self,
            "Changes Applied",
            f"Applied {len(internal_decisions)} internal duplicate decisions and "
            f"{len(external_decisions)} external duplicate decisions.",
        )

        # Enable Continue to Reports button
        self.continue_to_reports_btn.setEnabled(True)

    def reset_ui(self):
        """Reset the UI to its initial state."""
        self.internal_dup_df = None
        self.external_dup_df = None
        self.duplicate_groups = {}
        self.match_info = {}

        self.status_label.setText("Waiting for duplicate detection...")
        self.progress_bar.setValue(0)
        self.stats_label.setText(
            "Summary statistics will appear here after duplicate detection."
        )

        self.internal_table.setRowCount(0)
        self.external_table.setRowCount(0)

        self.resolve_selected_btn.setEnabled(False)
        self.resolve_all_btn.setEnabled(False)
        self.accept_all_matches_btn.setEnabled(False)
        self.reject_all_matches_btn.setEnabled(False)
        self.apply_changes_btn.setEnabled(False)
        self.continue_to_reports_btn.setEnabled(False)

        self.duplicate_status_changed.emit(False)

    def get_duplicate_results(self):
        """Get the duplicate detection results.

        Returns:
            tuple: (internal_dup_df, external_dup_df, duplicate_groups, match_info)
        """
        return (
            self.internal_dup_df,
            self.external_dup_df,
            self.duplicate_groups,
            self.match_info,
        )
