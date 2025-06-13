"""
report_frame.py

This module implements the Report Frame for the Entity Validation System.
It provides a UI for viewing and generating validation reports.
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
    QFileDialog,
    QListWidget,
    QListWidgetItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QIcon
import subprocess
import platform

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import validation modules
from validation.report_generation import ValidationReportGenerator


class ReportWorker(QThread):
    """Worker thread for generating reports in the background."""

    # Signals
    progress_updated = pyqtSignal(int)
    report_completed = pyqtSignal(dict)
    report_failed = pyqtSignal(str)

    def __init__(
        self,
        input_df,
        naming_results=None,
        duplicate_results=None,
        user_decisions=None,
        update_original=True,
    ):
        """Initialize the report worker.

        Args:
            input_df (pandas.DataFrame): Original input DataFrame.
            naming_results (list, optional): Naming convention results.
            duplicate_results (dict, optional): Duplicate detection results.
            user_decisions (dict, optional): User decisions on validation issues.
            update_original (bool, optional): Whether to update the original file.
        """
        super().__init__()
        self.input_df = input_df
        self.naming_results = naming_results
        self.duplicate_results = duplicate_results
        self.user_decisions = user_decisions
        self.update_original = update_original
        self.report_generator = None

    def run(self):
        """Run report generation in a separate thread."""
        try:
            # Initialize report generator
            self.report_generator = ValidationReportGenerator()

            # Check if dataframe is valid
            if self.input_df is None or self.input_df.empty:
                self.report_failed.emit("Input DataFrame is empty or invalid.")
                return

            # Start generating reports
            self.progress_updated.emit(25)

            # Generate all reports
            report_files = self.report_generator.generate_final_reports(
                self.input_df,
                self.naming_results,
                self.duplicate_results,
                self.user_decisions,
                self.update_original,
            )

            # Complete
            self.progress_updated.emit(100)
            self.report_completed.emit(report_files)

        except Exception as e:
            self.report_failed.emit(f"Report generation failed: {str(e)}")


class ReportFrame(QWidget):
    """Frame for viewing and generating validation reports."""

    # Signals
    report_status_changed = pyqtSignal(bool)  # True if report generation complete

    def __init__(self, parent=None):
        """Initialize the report frame."""
        super().__init__(parent)
        self.input_df = None
        self.naming_results = None
        self.duplicate_results = None
        self.user_decisions = None
        self.report_files = {}
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Status group
        status_group = QGroupBox("Report Generation Status")
        status_layout = QVBoxLayout(status_group)

        # Status label
        self.status_label = QLabel("No reports have been generated yet.")
        status_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        # Summary statistics
        self.stats_label = QLabel("Report summary will appear here after generation.")
        status_layout.addWidget(self.stats_label)

        main_layout.addWidget(status_group)

        # Report options group
        options_group = QGroupBox("Report Options")
        options_layout = QVBoxLayout(options_group)

        # Update original file option
        self.update_original_check = QCheckBox(
            "Update original file with validated entities"
        )
        self.update_original_check.setChecked(True)
        options_layout.addWidget(self.update_original_check)

        # Generate reports button
        self.generate_btn = QPushButton("Generate Reports")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.generate_reports)
        options_layout.addWidget(self.generate_btn)

        main_layout.addWidget(options_group)

        # Reports list group
        reports_group = QGroupBox("Generated Reports")
        reports_layout = QVBoxLayout(reports_group)

        # List widget for reports
        self.reports_list = QListWidget()
        self.reports_list.setAlternatingRowColors(True)
        self.reports_list.itemDoubleClicked.connect(self.open_report)
        reports_layout.addWidget(self.reports_list)

        # Report actions layout
        report_actions_layout = QHBoxLayout()

        # Open selected report button
        self.open_report_btn = QPushButton("Open Selected Report")
        self.open_report_btn.setEnabled(False)
        self.open_report_btn.clicked.connect(self.open_selected_report)
        report_actions_layout.addWidget(self.open_report_btn)

        # Open report folder button
        self.open_folder_btn = QPushButton("Open Reports Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self.open_reports_folder)
        report_actions_layout.addWidget(self.open_folder_btn)

        reports_layout.addLayout(report_actions_layout)

        main_layout.addWidget(reports_group)

    def set_data(
        self, input_df, naming_results=None, duplicate_results=None, user_decisions=None
    ):
        """Set data for report generation.

        Args:
            input_df (pandas.DataFrame): Original input DataFrame.
            naming_results (list, optional): Naming convention results.
            duplicate_results (dict, optional): Duplicate detection results.
            user_decisions (dict, optional): User decisions on validation issues.
        """
        if input_df is None or input_df.empty:
            QMessageBox.warning(self, "Invalid Data", "DataFrame is empty or invalid.")
            return

        # Store parameters
        self.input_df = input_df
        self.naming_results = naming_results
        self.duplicate_results = duplicate_results
        self.user_decisions = user_decisions

        # Enable generate button
        self.generate_btn.setEnabled(True)

        # Update status
        self.status_label.setText("Ready to generate reports.")

    def generate_reports(self):
        """Generate validation reports."""
        if self.input_df is None:
            QMessageBox.warning(
                self,
                "Cannot Generate Reports",
                "No data available for report generation.",
            )
            return

        # Reset UI
        self.reset_ui()

        # Get update original option
        update_original = self.update_original_check.isChecked()

        # Start report generation in worker thread
        self.worker = ReportWorker(
            self.input_df,
            self.naming_results,
            self.duplicate_results,
            self.user_decisions,
            update_original,
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.report_completed.connect(self.display_report_results)
        self.worker.report_failed.connect(self.handle_report_error)

        # Update status
        self.status_label.setText("Generating reports...")
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

    @pyqtSlot(dict)
    def display_report_results(self, report_files):
        """Display report generation results.

        Args:
            report_files (dict): Dictionary of generated report files.
        """
        self.report_files = report_files

        # Update status
        self.status_label.setText("Report generation complete.")
        self.progress_bar.setValue(100)

        # Calculate and display statistics
        num_reports = len(report_files)
        file_types = list(report_files.keys())

        self.stats_label.setText(
            f"Generated {num_reports} reports: {', '.join(file_types)}"
        )

        # Update reports list
        self.reports_list.clear()
        for report_type, file_path in report_files.items():
            item = QListWidgetItem(f"{report_type}: {os.path.basename(file_path)}")
            item.setData(Qt.UserRole, file_path)
            self.reports_list.addItem(item)

        # Enable action buttons
        self.open_report_btn.setEnabled(num_reports > 0)
        self.open_folder_btn.setEnabled(num_reports > 0)

        # Signal that report generation is complete
        self.report_status_changed.emit(True)

    @pyqtSlot(str)
    def handle_report_error(self, error_message):
        """Handle report generation errors.

        Args:
            error_message (str): Error message to display.
        """
        self.status_label.setText(f"Error: {error_message}")
        self.progress_bar.setValue(0)

        QMessageBox.critical(self, "Report Generation Error", error_message)

        # Signal that report generation failed
        self.report_status_changed.emit(False)

    def open_selected_report(self):
        """Open the selected report file with the default application."""
        selected_items = self.reports_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a report to open.")
            return

        # Get file path from selected item
        file_path = selected_items[0].data(Qt.UserRole)
        self.open_file(file_path)

    def open_report(self, item):
        """Open the report file associated with the double-clicked item.

        Args:
            item (QListWidgetItem): The item that was double-clicked.
        """
        file_path = item.data(Qt.UserRole)
        self.open_file(file_path)

    def open_file(self, file_path):
        """Open a file with the default application.

        Args:
            file_path (str): Path to the file to open.
        """
        if not os.path.exists(file_path):
            QMessageBox.warning(
                self, "File Not Found", f"The file no longer exists: {file_path}"
            )
            return

        try:
            # Open file with default application
            if platform.system() == "Windows":
                os.startfile(file_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", file_path], check=True)
            else:  # Linux
                subprocess.run(["xdg-open", file_path], check=True)

        except Exception as e:
            QMessageBox.critical(
                self, "Error Opening File", f"Could not open the file: {str(e)}"
            )

    def open_reports_folder(self):
        """Open the folder containing generated reports."""
        if not self.report_files:
            QMessageBox.warning(
                self, "No Reports", "No reports have been generated yet."
            )
            return

        # Get the directory of the first report
        report_dir = os.path.dirname(next(iter(self.report_files.values())))

        if not os.path.exists(report_dir):
            QMessageBox.warning(
                self,
                "Folder Not Found",
                f"The reports folder no longer exists: {report_dir}",
            )
            return

        try:
            # Open folder with default file explorer
            if platform.system() == "Windows":
                os.startfile(report_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", report_dir], check=True)
            else:  # Linux
                subprocess.run(["xdg-open", report_dir], check=True)

        except Exception as e:
            QMessageBox.critical(
                self, "Error Opening Folder", f"Could not open the folder: {str(e)}"
            )

    def reset_ui(self):
        """Reset the UI to its initial state."""
        self.report_files = {}

        self.status_label.setText("Waiting for report generation...")
        self.progress_bar.setValue(0)
        self.stats_label.setText("Report summary will appear here after generation.")

        self.reports_list.clear()

        self.open_report_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)

        self.report_status_changed.emit(False)

    def get_report_files(self):
        """Get the generated report files.

        Returns:
            dict: Dictionary of report files.
        """
        return self.report_files
