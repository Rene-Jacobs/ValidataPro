"""
main.py

This module implements the main interface and command-line functionality
for the Entity Validation System, integrating naming validation, duplicate
detection, and report generation.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import traceback
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QCheckBox,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QSlider,
)
from PyQt5.QtCore import Qt, QSettings, QSize, QPoint
from PyQt5.QtGui import QIcon, QFont

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GUI frames
from gui.validation_frame import ValidationFrame
from gui.duplicate_frame import DuplicateFrame
from gui.report_frame import ReportFrame

# Import validation modules
from validation.validation_rules import ValidationRules
from validation.naming_convention import NamingConventionValidator
from validation.duplicate_detection import DuplicateDetector
from validation.report_generation import ValidationReportGenerator

# Import utilities
from utils.file_handling import read_csv_file, standardize_columns, write_csv_file


# Configure logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up logging configuration.

    Args:
        log_level (int): Logging level (default: INFO)
        log_file (str, optional): Path to log file. If None, logs to console only.

    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"evs_{timestamp}.log")

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


class SetupFrame(QWidget):
    """Frame for configuration and file selection."""

    def __init__(self, parent=None):
        """Initialize the setup frame."""
        super().__init__(parent)
        self.logger = logging.getLogger(__name__ + ".SetupFrame")
        self.load_default_settings()

        # Find default AuthoritativeEntityList.csv
        self.default_auth_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "AuthoritativeEntityList.csv",
        )

        self.setup_ui()

        # Set default authoritative list if it exists
        if os.path.exists(self.default_auth_path):
            self.auth_file_edit.setText(self.default_auth_path)
            self.update_start_button()
            self.logger.info(
                f"Loaded default authoritative list: {self.default_auth_path}"
            )

    def load_default_settings(self):
        """Load default settings from validation_rules.py."""
        try:
            # Get default values from ValidationRules
            validation_rules = ValidationRules()
            self.default_threshold = int(validation_rules.DEFAULT_FUZZY_MATCH_THRESHOLD)
            self.logger.info(f"Loaded default threshold: {self.default_threshold}%")
        except Exception as e:
            self.logger.error(f"Error loading default settings: {str(e)}")
            self.default_threshold = 85  # Fallback default

    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)

        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)

        # Input file selection
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input File:"))
        self.input_file_edit = QLabel("No file selected")
        self.input_file_edit.setFrameStyle(QLabel.StyledPanel | QLabel.Sunken)
        input_layout.addWidget(self.input_file_edit, 1)
        self.input_file_btn = QPushButton("Browse...")
        self.input_file_btn.clicked.connect(self.browse_input_file)
        input_layout.addWidget(self.input_file_btn)
        file_layout.addLayout(input_layout)

        # Authoritative list selection
        auth_layout = QHBoxLayout()
        auth_layout.addWidget(QLabel("Authoritative List:"))
        self.auth_file_edit = QLabel("No file selected")
        self.auth_file_edit.setFrameStyle(QLabel.StyledPanel | QLabel.Sunken)
        auth_layout.addWidget(self.auth_file_edit, 1)
        self.auth_file_btn = QPushButton("Browse...")
        self.auth_file_btn.clicked.connect(self.browse_auth_file)
        auth_layout.addWidget(self.auth_file_btn)
        file_layout.addLayout(auth_layout)

        main_layout.addWidget(file_group)

        # Configuration group
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)

        # Entity column selection
        column_layout = QHBoxLayout()
        column_layout.addWidget(QLabel("Entity Column:"))
        self.column_combo = QComboBox()
        self.column_combo.addItem("name")
        self.column_combo.addItem("facility")
        self.column_combo.addItem("owner")
        self.column_combo.addItem("operator")
        column_layout.addWidget(self.column_combo)
        config_layout.addLayout(column_layout)

        # Fuzzy matching threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Matching Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(self.default_threshold)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        threshold_layout.addWidget(self.threshold_slider, 1)
        self.threshold_label = QLabel(f"{self.default_threshold}%")
        threshold_layout.addWidget(self.threshold_label)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        config_layout.addLayout(threshold_layout)

        # Advanced settings
        advanced_layout = QHBoxLayout()
        self.advanced_check = QCheckBox("Enable Advanced Matching")
        self.advanced_check.setToolTip(
            "Enables more sophisticated matching algorithms like ML-based matching"
        )
        advanced_layout.addWidget(self.advanced_check)
        config_layout.addLayout(advanced_layout)

        main_layout.addWidget(config_group)

        # Spacer
        main_layout.addStretch(1)

        # Start validation button
        start_layout = QHBoxLayout()
        start_layout.addStretch()
        self.start_btn = QPushButton("Start Validation")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_validation)
        start_layout.addWidget(self.start_btn)
        main_layout.addLayout(start_layout)

    def browse_input_file(self):
        """Browse for input CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            self.logger.info(f"Selected input file: {file_path}")
            self.input_file_edit.setText(file_path)
            self.update_start_button()

            # Try to read the file to get column names
            try:
                df = read_csv_file(file_path)
                df = standardize_columns(df)

                # Update column combo box
                self.column_combo.clear()
                for col in ["name", "facility", "owner", "operator"]:
                    if col in df.columns:
                        self.column_combo.addItem(col)

                # Default to "name" if available
                name_index = self.column_combo.findText("name")
                if name_index >= 0:
                    self.column_combo.setCurrentIndex(name_index)

                self.logger.info(f"Found columns: {list(df.columns)}")

            except Exception as e:
                self.logger.error(f"Error reading input file: {str(e)}")
                QMessageBox.warning(
                    self,
                    "Error Reading File",
                    f"Could not read the input file: {str(e)}",
                )

    def browse_auth_file(self):
        """Browse for authoritative entity list CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Authoritative Entity List",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )

        if file_path:
            self.logger.info(f"Selected authoritative list: {file_path}")
            self.auth_file_edit.setText(file_path)
            self.update_start_button()

    def update_threshold_label(self, value):
        """Update the threshold label when slider value changes."""
        self.threshold_label.setText(f"{value}%")
        self.logger.debug(f"Matching threshold updated to {value}%")

    def update_start_button(self):
        """Update the start button enabled state based on file selections."""
        input_file = self.input_file_edit.text()
        auth_file = self.auth_file_edit.text()

        enabled = (
            input_file != "No file selected"
            and auth_file != "No file selected"
            and os.path.exists(input_file)
            and os.path.exists(auth_file)
        )

        self.start_btn.setEnabled(enabled)

        if enabled:
            self.logger.debug("Validation ready to start")
        else:
            self.logger.debug("Validation not ready - missing files")

    def start_validation(self):
        """Start validation process."""
        # Validation will be handled by the parent window
        self.logger.info("Start validation button clicked")

    def get_input_file(self):
        """Get the selected input file path."""
        text = self.input_file_edit.text()
        return text if text != "No file selected" else None

    def get_auth_file(self):
        """Get the selected authoritative list file path."""
        text = self.auth_file_edit.text()
        return text if text != "No file selected" else None

    def get_entity_column(self):
        """Get the selected entity column."""
        return self.column_combo.currentText()

    def get_matching_threshold(self):
        """Get the selected matching threshold."""
        return self.threshold_slider.value() / 100.0

    def get_advanced_matching(self):
        """Get whether advanced matching is enabled."""
        return self.advanced_check.isChecked()


class MainWindow(QMainWindow):
    """Main window for the Entity Validation System."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".MainWindow")

        # Initialize data
        self.input_df = None
        self.naming_results = None
        self.duplicate_results = None
        self.match_info = None
        self.user_decisions = {}

        # Set up UI
        self.setup_ui()

        # Load settings
        self.load_settings()

        self.logger.info("MainWindow initialized")

    def setup_ui(self):
        """Set up the UI components."""
        # Window properties
        self.setWindowTitle("Entity Validation System")
        self.setMinimumSize(800, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Tab widget
        self.tab_widget = QTabWidget()

        # Setup tab
        self.setup_frame = SetupFrame()
        self.tab_widget.addTab(self.setup_frame, "Setup")

        # Validation tab
        self.validation_frame = ValidationFrame()
        self.validation_frame.validation_status_changed.connect(
            self.on_validation_status_changed
        )
        self.validation_frame.proceed_to_duplicates.connect(
            self.start_duplicate_detection
        )

        self.tab_widget.addTab(self.validation_frame, "Validation")

        # Duplicate tab
        self.duplicate_frame = DuplicateFrame()
        self.duplicate_frame.duplicate_status_changed.connect(
            self.on_duplicate_status_changed
        )
        self.duplicate_frame.continue_to_reports.connect(self.start_report_generation)
        self.tab_widget.addTab(self.duplicate_frame, "Duplicates")

        # Report tab
        self.report_frame = ReportFrame()
        self.report_frame.report_status_changed.connect(self.on_report_status_changed)
        self.tab_widget.addTab(self.report_frame, "Reports")

        main_layout.addWidget(self.tab_widget)

        # Connect setup frame start button
        self.setup_frame.start_btn.clicked.connect(self.start_validation_process)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def start_validation_process(self):
        """Start the validation process."""
        # Get parameters from setup frame
        input_file = self.setup_frame.get_input_file()
        auth_file = self.setup_frame.get_auth_file()
        entity_column = self.setup_frame.get_entity_column()
        matching_threshold = self.setup_frame.get_matching_threshold()
        advanced_matching = self.setup_frame.get_advanced_matching()

        self.logger.info(
            f"Starting validation process with parameters: input_file={input_file}, "
            f"auth_file={auth_file}, entity_column={entity_column}, "
            f"matching_threshold={matching_threshold}, advanced_matching={advanced_matching}"
        )

        # Check files
        if not input_file or not os.path.exists(input_file):
            self.logger.error(f"Invalid input file: {input_file}")
            QMessageBox.warning(
                self, "Invalid Input File", "Please select a valid input file."
            )
            return

        if not auth_file or not os.path.exists(auth_file):
            self.logger.error(f"Invalid authoritative list: {auth_file}")
            QMessageBox.warning(
                self,
                "Invalid Authoritative List",
                "Please select a valid authoritative entity list file.",
            )
            return

        try:
            # Load input file
            self.logger.info("Loading input file")
            self.input_df = read_csv_file(input_file)
            self.input_df = standardize_columns(self.input_df)

            # Check if entity column exists
            if entity_column not in self.input_df.columns:
                self.logger.error(
                    f"Entity column '{entity_column}' not found in input file"
                )
                QMessageBox.warning(
                    self,
                    "Invalid Column",
                    f"The selected entity column '{entity_column}' does not exist in the input file.",
                )
                return

            # Switch to validation tab
            self.tab_widget.setCurrentIndex(1)

            # Start validation
            self.logger.info("Starting naming convention validation")
            self.validation_frame.set_data(self.input_df, entity_column)
            self.status_bar.showMessage("Validation in progress...")

        except Exception as e:
            self.logger.error(f"Error in validation process: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(
                self, "Error Loading File", f"Could not load the input file: {str(e)}"
            )

    def on_validation_status_changed(self, completed):
        """Handle validation status changes.

        Args:
            completed (bool): Whether validation completed successfully.
        """
        if completed:
            self.logger.info("Validation completed successfully")
            self.status_bar.showMessage("Validation completed.")

            # Store validation results
            self.naming_results = self.validation_frame.get_validation_results()

            # Collect user decisions
            if self.naming_results:
                decisions = []
                for result in self.naming_results:
                    if result["status"] in ["accepted", "rejected", "modified"]:
                        decisions.append(
                            {
                                "entity": result["entity"],
                                "action": result["status"],
                                "modified_value": (
                                    result["suggestion"]
                                    if result["status"] == "modified"
                                    else None
                                ),
                            }
                        )

                if decisions:
                    self.user_decisions["naming_convention"] = decisions
                    self.logger.info(
                        f"Collected {len(decisions)} naming convention decisions"
                    )

            # Enable duplicate tab
            self.tab_widget.setTabEnabled(2, True)

            # Show notification
            QMessageBox.information(
                self,
                "Validation Complete",
                "Naming convention validation is complete. You can now proceed to duplicate detection.",
            )

        else:
            self.logger.warning("Validation failed or canceled")
            self.status_bar.showMessage("Validation failed or canceled.")

    def start_duplicate_detection(self):
        """Start the duplicate detection process."""
        # Get parameters from setup frame
        auth_file = self.setup_frame.get_auth_file()
        matching_threshold = self.setup_frame.get_matching_threshold()
        advanced_matching = self.setup_frame.get_advanced_matching()

        self.logger.info(
            f"Starting duplicate detection with threshold={matching_threshold}, "
            f"advanced_matching={advanced_matching}"
        )

        # Apply naming convention corrections if available
        if self.validation_frame.get_validated_dataframe() is not None:
            self.input_df = self.validation_frame.get_validated_dataframe()
            self.logger.info("Using validated DataFrame for duplicate detection")

        # Switch to duplicate tab
        self.tab_widget.setCurrentIndex(2)

        # Start duplicate detection
        self.duplicate_frame.set_data(self.input_df, auth_file, matching_threshold)
        self.status_bar.showMessage("Duplicate detection in progress...")

    def on_duplicate_status_changed(self, completed):
        """Handle duplicate detection status changes.

        Args:
            completed (bool): Whether duplicate detection completed successfully.
        """
        if completed:
            self.logger.info("Duplicate detection completed successfully")
            self.status_bar.showMessage("Duplicate detection completed.")

            # Store duplicate results
            result_tuple = self.duplicate_frame.get_duplicate_results()
            (
                self.internal_dup_df,
                self.external_dup_df,
                self.duplicate_groups,
                self.match_info,
            ) = result_tuple

            # Enable report tab
            self.tab_widget.setTabEnabled(3, True)

            # Show notification that duplicate detection is complete and user should
            # review duplicates and then click "Continue to Reports" when ready
            QMessageBox.information(
                self,
                "Duplicate Detection Complete",
                "Duplicate detection is complete. Please review the potential duplicates "
                "and make decisions on each one. When you're finished, click 'Continue to Reports' "
                "to generate validation reports.",
            )

            # No automatic prompt to move to report generation -
            # user must click the "Continue to Reports" button when ready
        else:
            self.logger.warning("Duplicate detection failed or canceled")
            self.status_bar.showMessage("Duplicate detection failed or canceled.")

    def on_report_status_changed(self, completed):
        """Handle report generation status changes.

        Args:
            completed (bool): Whether report generation completed successfully.
        """
        if completed:
            self.logger.info("Report generation completed successfully")
            self.status_bar.showMessage("Report generation completed.")

            # Get report files
            report_files = self.report_frame.get_report_files()
            self.logger.info(f"Generated reports: {list(report_files.keys())}")

            # Save settings
            self.save_settings()
        else:
            self.logger.warning("Report generation failed or canceled")
            self.status_bar.showMessage("Report generation failed or canceled.")

    def start_report_generation(self):
        """Start the report generation process."""
        self.logger.info("Starting report generation")

        # Collect user decisions for duplicate detection
        # For internal duplicates
        internal_decisions = {}
        # This would normally collect internal duplicate decisions from UI
        self.user_decisions["internal_duplicates"] = internal_decisions

        # For external duplicates
        external_decisions = {}
        # This would normally collect external duplicate decisions from UI
        self.user_decisions["external_duplicates"] = external_decisions

        # Switch to report tab
        self.tab_widget.setCurrentIndex(3)

        # Set data for report generation
        self.report_frame.set_data(
            self.input_df, self.naming_results, self.match_info, self.user_decisions
        )
        self.status_bar.showMessage("Ready to generate reports.")

    def load_settings(self):
        """Load application settings."""
        settings = QSettings("EVS", "EntityValidationSystem")

        # Window geometry
        settings.setValue("geometry", self.saveGeometry())

        # Tab state
        settings.setValue("tab_index", self.tab_widget.currentIndex())

        # Save validation rules settings
        settings.setValue(
            "matching_threshold", self.setup_frame.get_matching_threshold()
        )
        settings.setValue("advanced_matching", self.setup_frame.get_advanced_matching())

        self.logger.info("Saved application settings")

    def center_on_screen(self):
        """Center the window on the screen."""
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.desktop().screenGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

    def closeEvent(self, event):
        """Handle window close event to save settings."""
        self.logger.info("Application closing, saving settings")
        self.save_settings()
        super().closeEvent(event)


def run_batch_mode(args):
    """
    Run the application in batch mode without GUI.

    Args:
        args (argparse.Namespace): Command-line arguments

    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__ + ".batch_mode")
    logger.info("Starting batch processing mode")

    try:
        # Load input file
        logger.info(f"Loading input file: {args.input}")
        input_df = read_csv_file(args.input)
        input_df = standardize_columns(input_df)

        # Check entity column
        if args.column not in input_df.columns:
            logger.error(f"Entity column '{args.column}' not found in input file")
            return False

        # Step 1: Naming convention validation
        logger.info("Starting naming convention validation")
        validator = NamingConventionValidator()
        naming_results, validated_df = validator.validate_dataframe(
            input_df, args.column
        )

        logger.info(f"Validated {len(naming_results)} entities")

        # Step 2: Duplicate detection
        logger.info(f"Starting duplicate detection with threshold {args.threshold}")
        detector = DuplicateDetector(args.auth_list, args.threshold)

        # Detect internal duplicates
        internal_dup_df, duplicate_groups = detector.detect_internal_duplicates(
            validated_df
        )

        # Detect duplicates against authoritative list
        external_dup_df, match_info = detector.detect_duplicates(internal_dup_df)

        logger.info(f"Found {len(duplicate_groups)} internal duplicate groups")
        external_matches = sum(
            1
            for entity_info in match_info.values()
            if entity_info and any(m.get("best_match") for m in entity_info.values())
        )
        logger.info(
            f"Found {external_matches} potential matches with authoritative list"
        )

        # Step 3: Generate reports
        logger.info("Generating reports")
        report_generator = ValidationReportGenerator(args.output_dir)

        # Generate all reports
        report_files = report_generator.generate_final_reports(
            input_df,
            naming_results,
            match_info,
            None,  # No user decisions in batch mode
            args.update_original,
        )

        logger.info(f"Generated reports: {list(report_files.keys())}")

        # Print summary
        total_entities = len(input_df)
        valid_entities = sum(1 for r in naming_results if r["valid"])
        invalid_entities = total_entities - valid_entities

        print("\nProcessing Summary:")
        print(f"Total entities processed: {total_entities}")
        print(f"Valid naming conventions: {valid_entities}")
        print(f"Invalid naming conventions: {invalid_entities}")
        print(f"Internal duplicate groups: {len(duplicate_groups)}")
        print(f"External matches: {external_matches}")
        print("\nGenerated reports:")
        for report_type, file_path in report_files.items():
            print(f"- {report_type}: {file_path}")

        return True

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        print(f"Error: {str(e)}")
        return False


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Entity Validation System")

    # Mode selection
    parser.add_argument(
        "--batch", action="store_true", help="Run in batch mode without GUI"
    )

    # File paths
    parser.add_argument("--input", help="Input CSV file path")
    parser.add_argument("--auth-list", help="Authoritative entity list CSV file path")
    parser.add_argument("--output-dir", help="Output directory for reports")

    # Configuration
    parser.add_argument(
        "--column", default="name", help="Entity column name (default: name)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Matching threshold (0-1, default: 0.85)",
    )
    parser.add_argument(
        "--update-original",
        action="store_true",
        help="Update original file with corrections",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument("--log-file", help="Log file path (default: auto-generated)")

    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level, args.log_file)
    logger.info("Entity Validation System starting")

    # Run in batch mode if requested
    if args.batch:
        if not args.input or not args.auth_list:
            print("Error: --input and --auth-list are required for batch mode")
            return 1

        # Set default output directory if not specified
        if not args.output_dir:
            args.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
            )

        # Run batch processing
        success = run_batch_mode(args)
        return 0 if success else 1

    # Run in GUI mode
    try:
        app = QApplication(sys.argv)

        # Set application style
        app.setStyle("Fusion")

        # Create and show main window
        main_window = MainWindow()
        main_window.show()

        # Run application
        logger.info("Starting GUI application")
        return app.exec_()

    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        print(f"Application crashed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

    # Window geometry
    geometry = settings.value("geometry")
    if geometry:
        self.restoreGeometry(geometry)
        self.logger.debug("Restored window geometry from settings")
    else:
        # Default size and position
        self.resize(900, 700)
        self.center_on_screen()
        self.logger.debug("Using default window size and position")

    # Tab state
    tab_index = settings.value("tab_index", 0, type=int)
    self.tab_widget.setCurrentIndex(tab_index)
    self.logger.debug(f"Set current tab to {tab_index}")

    # Load validation rules settings
    try:
        threshold = settings.value("matching_threshold", None)
        if threshold is not None:
            self.setup_frame.threshold_slider.setValue(int(float(threshold) * 100))
            self.logger.debug(f"Loaded matching threshold from settings: {threshold}")

        advanced_matching = settings.value("advanced_matching", False, type=bool)
        self.setup_frame.advanced_check.setChecked(advanced_matching)
        self.logger.debug(f"Loaded advanced matching setting: {advanced_matching}")

    except Exception as e:
        self.logger.error(f"Error loading settings: {str(e)}")

    # Disable tabs until validation is done
    self.tab_widget.setTabEnabled(1, False)
    self.tab_widget.setTabEnabled(2, False)
    self.tab_widget.setTabEnabled(3, False)

    def save_settings(self):
        """Save application settings."""
        settings = QSettings("EVS", "EntityValidationSystem")
