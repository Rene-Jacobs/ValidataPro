"""
validation_frame.py

This module implements the Validation Frame for the Entity Validation System.
It provides a UI for reviewing and managing naming convention validation results.
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
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QIcon

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import validation modules
from validation.validation_rules import ValidationRules
from validation.naming_convention import NamingConventionValidator


class ValidationWorker(QThread):
    """Worker thread for running validation operations in the background."""

    # Signals
    progress_updated = pyqtSignal(int)
    validation_completed = pyqtSignal(list, object)
    validation_failed = pyqtSignal(str)

    def __init__(self, df, entity_column="name"):
        """Initialize the validation worker.

        Args:
            df (pandas.DataFrame): DataFrame with entity data.
            entity_column (str, optional): Column containing entity names.
                Defaults to 'name'.
        """
        super().__init__()
        self.df = df
        self.entity_column = entity_column
        self.validator = None

    def run(self):
        """Run validation in a separate thread."""
        try:
            # Initialize validator
            self.validator = NamingConventionValidator()

            # Get list of entities to validate
            entities = self.df[self.entity_column].dropna().astype(str).tolist()
            total_entities = len(entities)

            if total_entities == 0:
                self.validation_failed.emit("No entities found to validate.")
                return

            # Validate entities
            results = []
            for i, entity in enumerate(entities):
                result = self.validator.validate_entity(entity)
                results.append(result)

                # Update progress
                progress = int((i + 1) / total_entities * 100)
                self.progress_updated.emit(progress)

            # Validate entire dataframe
            _, validated_df = self.validator.validate_dataframe(
                self.df, self.entity_column
            )

            # Signal completion
            self.validation_completed.emit(results, validated_df)

        except Exception as e:
            self.validation_failed.emit(f"Validation failed: {str(e)}")


class ValidationFrame(QWidget):
    """Frame for displaying and managing naming convention validation results."""

    # Signals
    validation_status_changed = pyqtSignal(bool)  # True if validation complete
    proceed_to_duplicates = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the validation frame."""
        super().__init__(parent)
        self.validation_results = []
        self.validated_df = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Status group
        status_group = QGroupBox("Validation Status")
        status_layout = QVBoxLayout(status_group)

        # Status label
        self.status_label = QLabel("No file has been validated yet.")
        status_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        # Summary statistics
        self.stats_label = QLabel(
            "Summary statistics will appear here after validation."
        )
        status_layout.addWidget(self.stats_label)

        main_layout.addWidget(status_group)

        # Results table
        results_group = QGroupBox("Naming Convention Violations")
        results_layout = QVBoxLayout(results_group)

        # Set up table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Entity", "Violations", "Suggestion", "Action", "Status"]
        )

        # Set all columns to be interactive (manually resizable)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )

        # Set default widths that make sense as a starting point
        self.results_table.setColumnWidth(0, 200)  # Entity
        self.results_table.setColumnWidth(1, 250)  # Violations
        self.results_table.setColumnWidth(2, 200)  # Suggestion
        self.results_table.setColumnWidth(3, 100)  # Action
        self.results_table.setColumnWidth(4, 100)  # Status

        # Make the Action column fixed width since it contains combo boxes
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)

        # Enable alternating row colors for better readability
        self.results_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.results_table)

        # Action buttons
        action_layout = QHBoxLayout()

        self.accept_all_btn = QPushButton("Accept All")
        self.accept_all_btn.setEnabled(False)
        self.accept_all_btn.clicked.connect(self.accept_all_suggestions)
        action_layout.addWidget(self.accept_all_btn)

        self.reject_all_btn = QPushButton("Reject All")
        self.reject_all_btn.setEnabled(False)
        self.reject_all_btn.clicked.connect(self.reject_all_suggestions)
        action_layout.addWidget(self.reject_all_btn)
        results_layout.addLayout(action_layout)

        proceed_layout = QHBoxLayout()
        self.proceed_to_duplicates_btn = QPushButton("Proceed to Duplicate Detection")
        self.proceed_to_duplicates_btn.setEnabled(False)
        self.proceed_to_duplicates_btn.clicked.connect(self.on_proceed_to_duplicates)
        proceed_layout.addWidget(self.proceed_to_duplicates_btn)
        results_layout.addLayout(proceed_layout)

        main_layout.addWidget(results_group)

        # Buttons layout at bottom
        buttons_layout = QHBoxLayout()

        # Apply changes button
        self.apply_changes_btn = QPushButton("Apply Changes")
        self.apply_changes_btn.setEnabled(False)
        self.apply_changes_btn.clicked.connect(self.apply_changes)
        buttons_layout.addWidget(self.apply_changes_btn)

        main_layout.addLayout(buttons_layout)

    def set_data(self, df, entity_column="name"):
        """Set data for validation.

        Args:
            df (pandas.DataFrame): DataFrame to validate.
            entity_column (str, optional): Column containing entity names.
                Defaults to 'name'.
        """
        if df is None or df.empty or entity_column not in df.columns:
            QMessageBox.warning(
                self,
                "Invalid Data",
                f"DataFrame is empty or does not contain the '{entity_column}' column.",
            )
            return

        # Reset UI
        self.reset_ui()

        # Start validation in worker thread
        self.worker = ValidationWorker(df, entity_column)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.validation_completed.connect(self.display_validation_results)
        self.worker.validation_failed.connect(self.handle_validation_error)

        # Update status
        self.status_label.setText("Validating entities...")
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

    @pyqtSlot(list, object)
    def display_validation_results(self, results, validated_df):
        """Display validation results in the table.

        Args:
            results (list): List of validation result dictionaries.
            validated_df (pandas.DataFrame): DataFrame with validation columns.
        """
        self.validation_results = results
        self.validated_df = validated_df

        # Update status
        self.status_label.setText("Validation complete.")
        self.progress_bar.setValue(100)

        # Calculate and display statistics
        total_entities = len(results)
        valid_entities = sum(1 for r in results if r["valid"])
        invalid_entities = total_entities - valid_entities

        self.stats_label.setText(
            f"Total entities: {total_entities}, "
            f"Valid: {valid_entities}, "
            f"Invalid: {invalid_entities}"
        )

        # Only show invalid entities in the table
        invalid_results = [r for r in results if not r["valid"]]

        # Update table
        self.results_table.setRowCount(len(invalid_results))

        for i, result in enumerate(invalid_results):
            # Entity name
            entity_item = QTableWidgetItem(result["entity"])
            entity_item.setFlags(entity_item.flags() & ~Qt.ItemIsEditable)
            self.results_table.setItem(i, 0, entity_item)

            # Violations
            violations_text = "; ".join(result["violations"])
            violations_item = QTableWidgetItem(violations_text)
            violations_item.setFlags(violations_item.flags() & ~Qt.ItemIsEditable)
            self.results_table.setItem(i, 1, violations_item)

            # Suggestion
            suggestion_item = QTableWidgetItem(result["suggestion"])
            suggestion_item.setFlags(suggestion_item.flags() | Qt.ItemIsEditable)
            self.results_table.setItem(i, 2, suggestion_item)

            # Action combo box
            action_widget = QComboBox()
            action_widget.addItems(["Choose Action", "Accept", "Reject", "Modify"])
            action_widget.setCurrentIndex(0)
            action_widget.setProperty("row", i)
            action_widget.currentIndexChanged.connect(self.on_action_changed)
            self.results_table.setCellWidget(i, 3, action_widget)

            # Status
            status_item = QTableWidgetItem(result["status"].capitalize())
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            self.results_table.setItem(i, 4, status_item)

            # Set row color based on status
            if result["status"] == "accepted":
                self.set_row_color(i, QColor(200, 255, 200))  # Light green
            elif result["status"] == "rejected":
                self.set_row_color(i, QColor(255, 200, 200))  # Light red
            elif result["status"] == "modified":
                self.set_row_color(i, QColor(200, 200, 255))  # Light blue

        # Enable action buttons if there are results
        if invalid_results:
            self.accept_all_btn.setEnabled(True)
            self.reject_all_btn.setEnabled(True)
            self.apply_changes_btn.setEnabled(True)
            self.proceed_to_duplicates_btn.setEnabled(True)

        # Signal that validation is complete
        self.validation_status_changed.emit(True)

    @pyqtSlot(str)
    def handle_validation_error(self, error_message):
        """Handle validation errors.

        Args:
            error_message (str): Error message to display.
        """
        self.status_label.setText(f"Error: {error_message}")
        self.progress_bar.setValue(0)

        QMessageBox.critical(self, "Validation Error", error_message)

        # Signal that validation failed
        self.validation_status_changed.emit(False)

    def on_action_changed(self, index):
        """Handle action combo box changes.

        Args:
            index (int): Selected index in the combo box.
        """
        combo_box = self.sender()
        row = combo_box.property("row")

        # Get the original index in validation_results
        entity = self.results_table.item(row, 0).text()
        result_index = next(
            (i for i, r in enumerate(self.validation_results) if r["entity"] == entity),
            -1,
        )

        if result_index == -1:
            return

        if index == 1:  # Accept
            # Update status
            self.validation_results[result_index]["status"] = "accepted"
            status_item = QTableWidgetItem("Accepted")
            self.results_table.setItem(row, 4, status_item)

            # Update row color
            self.set_row_color(row, QColor(200, 255, 200))  # Light green

        elif index == 2:  # Reject
            # Update status
            self.validation_results[result_index]["status"] = "rejected"
            status_item = QTableWidgetItem("Rejected")
            self.results_table.setItem(row, 4, status_item)

            # Update row color
            self.set_row_color(row, QColor(255, 200, 200))  # Light red

        elif index == 3:  # Modify
            # Update status
            self.validation_results[result_index]["status"] = "modified"

            # Get the suggestion
            suggestion = self.results_table.item(row, 2).text()
            self.validation_results[result_index]["suggestion"] = suggestion

            status_item = QTableWidgetItem("Modified")
            self.results_table.setItem(row, 4, status_item)

            # Update row color
            self.set_row_color(row, QColor(200, 200, 255))  # Light blue

    def set_row_color(self, row, color):
        """Set the background color for a table row.

        Args:
            row (int): Row index.
            color (QColor): Background color.
        """
        for col in range(self.results_table.columnCount()):
            item = self.results_table.item(row, col)
            if item:
                item.setBackground(color)

    def accept_all_suggestions(self):
        """Accept all validation suggestions."""
        for i in range(self.results_table.rowCount()):
            # Get entity name
            entity = self.results_table.item(i, 0).text()

            # Find result index
            result_index = next(
                (
                    i
                    for i, r in enumerate(self.validation_results)
                    if r["entity"] == entity
                ),
                -1,
            )

            if result_index != -1:
                # Update status
                self.validation_results[result_index]["status"] = "accepted"

                # Update table
                status_item = QTableWidgetItem("Accepted")
                self.results_table.setItem(i, 4, status_item)

                # Set action combo box
                action_widget = self.results_table.cellWidget(i, 3)
                if action_widget:
                    action_widget.setCurrentIndex(1)  # Accept

                # Update row color
                self.set_row_color(i, QColor(200, 255, 200))  # Light green

    def reject_all_suggestions(self):
        """Reject all validation suggestions."""
        for i in range(self.results_table.rowCount()):
            # Get entity name
            entity = self.results_table.item(i, 0).text()

            # Find result index
            result_index = next(
                (
                    i
                    for i, r in enumerate(self.validation_results)
                    if r["entity"] == entity
                ),
                -1,
            )

            if result_index != -1:
                # Update status
                self.validation_results[result_index]["status"] = "rejected"

                # Update table
                status_item = QTableWidgetItem("Rejected")
                self.results_table.setItem(i, 4, status_item)

                # Set action combo box
                action_widget = self.results_table.cellWidget(i, 3)
                if action_widget:
                    action_widget.setCurrentIndex(2)  # Reject

                # Update row color
                self.set_row_color(i, QColor(255, 200, 200))  # Light red

    def apply_changes(self):
        """Apply validation changes to the dataframe."""
        if not self.validation_results or not self.validated_df is not None:
            QMessageBox.warning(
                self, "Cannot Apply Changes", "No validation results available."
            )
            return

        # Process user decisions
        decisions = []
        for result in self.validation_results:
            if result["status"] in ["accepted", "rejected", "modified"]:
                decision = {
                    "entity": result["entity"],
                    "action": result["status"],
                }

                if result["status"] == "modified":
                    decision["modified_value"] = result["suggestion"]

                decisions.append(decision)

        # Create NamingConventionValidator instance
        validator = NamingConventionValidator()

        # Process user decisions
        results = validator.process_user_decisions(self.validation_results, decisions)

        # Show confirmation
        QMessageBox.information(
            self,
            "Changes Applied",
            f"{len(decisions)} validation decisions have been applied.",
        )

    def on_proceed_to_duplicates(self):
        """Handle click on Proceed to Duplicate Detection button."""
        self.proceed_to_duplicates.emit()

    def reset_ui(self):
        """Reset the UI to its initial state."""
        self.validation_results = []
        self.validated_df = None

        self.status_label.setText("Waiting for validation...")
        self.progress_bar.setValue(0)
        self.stats_label.setText(
            "Summary statistics will appear here after validation."
        )

        self.results_table.setRowCount(0)

        self.accept_all_btn.setEnabled(False)
        self.reject_all_btn.setEnabled(False)
        self.apply_changes_btn.setEnabled(False)
        self.proceed_to_duplicates_btn.setEnabled(False)

        self.validation_status_changed.emit(False)

    def get_validation_results(self):
        """Get the validation results.

        Returns:
            list: List of validation result dictionaries.
        """
        return self.validation_results

    def get_validated_dataframe(self):
        """Get the validated DataFrame.

        Returns:
            pandas.DataFrame: DataFrame with validation columns.
        """
        return self.validated_df
