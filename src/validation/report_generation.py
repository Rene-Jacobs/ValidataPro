"""
report_generation.py

This module generates detailed validation reports for entity data, including
naming convention violations, duplicate matches, and unresolved issues.
"""

import os
import sys
import pandas as pd
import logging
import csv
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from src.utils.file_handling import write_csv_file

# Configure logging
logger = logging.getLogger("report_generation")


class ValidationReportGenerator:
    """
    Class for generating detailed validation reports based on entity validation results.
    """

    def __init__(self, output_dir=None):
        """
        Initialize the ValidationReportGenerator.

        Args:
            output_dir (str, optional): Directory to save report files.
                Defaults to None, which uses a 'reports' directory in the project root.
        """
        # Set default output directory if not provided
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.output_dir = os.path.join(base_dir, "reports")
        else:
            self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize audit log
        self.audit_log = []

    def _create_timestamp(self):
        """
        Create a timestamp string for file naming.

        Returns:
            str: Timestamp in YYYYMMDD_HHMMSS format.
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_validation_action(self, action_type, entity_name, details=None):
        """
        Log a validation action to the audit log.

        Args:
            action_type (str): Type of action (e.g., 'naming_convention', 'duplicate').
            entity_name (str): Name of the entity involved.
            details (dict, optional): Additional details about the action.
        """
        timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "action_type": action_type,
            "entity_name": entity_name,
            "details": details or {},
        }

        self.audit_log.append(log_entry)
        logger.info(f"Logged {action_type} action for entity '{entity_name}'")

    def export_audit_log(self, filename=None):
        """
        Export the audit log to a CSV file.

        Args:
            filename (str, optional): Name of the file to save.
                Defaults to None, which generates a name with timestamp.

        Returns:
            str: Path to the exported audit log file.
        """
        if not self.audit_log:
            logger.warning("No audit log entries to export")
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = self._create_timestamp()
            filename = f"audit_log_{timestamp}.csv"

        # Create full path
        file_path = os.path.join(self.output_dir, filename)

        try:
            # Flatten the log entries for CSV export
            flattened_log = []
            for entry in self.audit_log:
                flat_entry = {
                    "timestamp": entry["timestamp"],
                    "action_type": entry["action_type"],
                    "entity_name": entry["entity_name"],
                }

                # Add details as separate columns
                if entry["details"]:
                    for key, value in entry["details"].items():
                        if isinstance(value, (dict, list)):
                            flat_entry[key] = json.dumps(value)
                        else:
                            flat_entry[key] = value

                flattened_log.append(flat_entry)

            # Convert to DataFrame and export
            audit_df = pd.DataFrame(flattened_log)
            write_csv_file(audit_df, file_path)

            logger.info(f"Exported audit log to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error exporting audit log: {str(e)}")
            return None

    def generate_naming_convention_report(self, validation_results, filename=None):
        """
        Generate a report of naming convention violations and corrections.

        Args:
            validation_results (List[Dict]): Results from naming_convention.py.
            filename (str, optional): Name of the file to save.

        Returns:
            str: Path to the generated report file.
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = self._create_timestamp()
            filename = f"naming_convention_report_{timestamp}.csv"

        # Create full path
        file_path = os.path.join(self.output_dir, filename)

        try:
            # Prepare data for report
            report_data = []
            for result in validation_results:
                if not result["valid"]:
                    report_entry = {
                        "entity": result["entity"],
                        "violations": "; ".join(result["violations"]),
                        "suggestion": result["suggestion"],
                        "status": result["status"],
                    }
                    report_data.append(report_entry)

                    # Log to audit log
                    self.log_validation_action(
                        "naming_convention",
                        result["entity"],
                        {
                            "violations": result["violations"],
                            "suggestion": result["suggestion"],
                            "status": result["status"],
                        },
                    )

            # Create and export DataFrame
            if report_data:
                report_df = pd.DataFrame(report_data)
                write_csv_file(report_df, file_path)
                logger.info(
                    f"Generated naming convention report with {len(report_data)} entries: {file_path}"
                )
                return file_path
            else:
                logger.info("No naming convention violations to report")
                return None

        except Exception as e:
            logger.error(f"Error generating naming convention report: {str(e)}")
            return None

    def generate_duplicate_detection_report(
        self, duplicate_results, user_decisions=None, filename=None
    ):
        """
        Generate a report of duplicate detection results and user decisions.

        Args:
            duplicate_results (Dict): Results from duplicate_detection.py.
            user_decisions (Dict, optional): User decisions on duplicates.
            filename (str, optional): Name of the file to save.

        Returns:
            str: Path to the generated report file.
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = self._create_timestamp()
            filename = f"duplicate_detection_report_{timestamp}.csv"

        # Create full path
        file_path = os.path.join(self.output_dir, filename)

        try:
            # Prepare data for report
            report_data = []

            for entity, result in duplicate_results.items():
                # Get match information
                best_match = result.get("best_match")
                best_score = result.get("best_score", 0)
                is_exact_match = result.get("is_exact_match", False)
                is_new_entity = result.get("is_new_entity", True)

                # Get user decision for this entity if available
                user_decision = None
                if user_decisions and entity in user_decisions:
                    user_decision = user_decisions[entity].get("action")

                # Create report entry
                report_entry = {
                    "entity": entity,
                    "best_match": best_match,
                    "match_score": best_score,
                    "is_exact_match": is_exact_match,
                    "is_new_entity": is_new_entity,
                    "user_decision": user_decision,
                }

                # Add all matches if available
                matches = result.get("matches", [])
                if matches:
                    match_details = "; ".join(
                        [f"{match} ({score:.3f})" for match, score in matches]
                    )
                    report_entry["all_matches"] = match_details

                report_data.append(report_entry)

                # Log to audit log
                self.log_validation_action(
                    "duplicate_detection",
                    entity,
                    {
                        "best_match": best_match,
                        "match_score": best_score,
                        "is_exact_match": is_exact_match,
                        "is_new_entity": is_new_entity,
                        "user_decision": user_decision,
                    },
                )

            # Create and export DataFrame
            if report_data:
                report_df = pd.DataFrame(report_data)
                write_csv_file(report_df, file_path)
                logger.info(
                    f"Generated duplicate detection report with {len(report_data)} entries: {file_path}"
                )
                return file_path
            else:
                logger.info("No duplicates to report")
                return None

        except Exception as e:
            logger.error(f"Error generating duplicate detection report: {str(e)}")
            return None

    def generate_unresolved_issues_report(self, validation_df, filename=None):
        """
        Generate a report of unresolved validation issues.

        Args:
            validation_df (pandas.DataFrame): DataFrame with validation results.
            filename (str, optional): Name of the file to save.

        Returns:
            str: Path to the generated report file.
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = self._create_timestamp()
            filename = f"unresolved_issues_report_{timestamp}.csv"

        # Create full path
        file_path = os.path.join(self.output_dir, filename)

        try:
            # Filter for unresolved issues
            unresolved_df = validation_df.copy()

            # Check for naming convention validation columns
            if all(
                col in unresolved_df.columns
                for col in ["validation_valid", "validation_status"]
            ):
                naming_mask = (~unresolved_df["validation_valid"]) & (
                    unresolved_df["validation_status"].isin(["pending", "rejected"])
                )
            else:
                naming_mask = pd.Series([False] * len(unresolved_df))

            # Check for duplicate detection columns
            if "is_duplicate" in unresolved_df.columns:
                # Unresolved duplicates are those that are flagged but not exact matches
                dup_mask = unresolved_df["is_duplicate"] & (
                    ~unresolved_df.get("is_exact_match", False)
                )
            else:
                dup_mask = pd.Series([False] * len(unresolved_df))

            # Combined mask for all unresolved issues
            unresolved_mask = naming_mask | dup_mask

            # Filter DataFrame
            unresolved_df = unresolved_df[unresolved_mask]

            # If no unresolved issues, return None
            if unresolved_df.empty:
                logger.info("No unresolved issues to report")
                return None

            # Select relevant columns
            relevant_cols = []

            # Original data columns
            entity_cols = [
                col
                for col in ["name", "facility", "owner", "operator"]
                if col in unresolved_df.columns
            ]
            relevant_cols.extend(entity_cols)

            # Naming convention columns
            naming_cols = [
                col
                for col in [
                    "validation_valid",
                    "validation_violations",
                    "validation_suggestion",
                    "validation_status",
                ]
                if col in unresolved_df.columns
            ]
            relevant_cols.extend(naming_cols)

            # Duplicate detection columns
            dup_cols = [
                col
                for col in [
                    "is_duplicate",
                    "duplicate_match",
                    "duplicate_score",
                    "is_exact_match",
                ]
                if col in unresolved_df.columns
            ]
            relevant_cols.extend(dup_cols)

            # Export filtered DataFrame
            result_df = unresolved_df[relevant_cols]
            write_csv_file(result_df, file_path)

            logger.info(
                f"Generated unresolved issues report with {len(result_df)} entries: {file_path}"
            )

            # Log to audit log
            self.log_validation_action(
                "unresolved_issues",
                "multiple_entities",
                {
                    "issue_count": len(result_df),
                    "naming_convention_issues": int(naming_mask.sum()),
                    "duplicate_issues": int(dup_mask.sum()),
                },
            )

            return file_path

        except Exception as e:
            logger.error(f"Error generating unresolved issues report: {str(e)}")
            return None

    def generate_comprehensive_report(
        self,
        input_df,
        naming_results=None,
        duplicate_results=None,
        user_decisions=None,
        filename=None,
    ):
        """
        Generate a comprehensive validation report with all issues.

        Args:
            input_df (pandas.DataFrame): Original input DataFrame.
            naming_results (List[Dict], optional): Naming convention results.
            duplicate_results (Dict, optional): Duplicate detection results.
            user_decisions (Dict, optional): User decisions on validation issues.
            filename (str, optional): Name of the file to save.

        Returns:
            str: Path to the generated report file.
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = self._create_timestamp()
            filename = f"comprehensive_report_{timestamp}.csv"

        # Create full path
        file_path = os.path.join(self.output_dir, filename)

        try:
            # Start with a copy of the input DataFrame
            report_df = input_df.copy()

            # Add report columns
            report_df["has_naming_issues"] = False
            report_df["naming_violations"] = None
            report_df["naming_suggestion"] = None
            report_df["naming_status"] = None

            report_df["has_duplicate_issues"] = False
            report_df["duplicate_match"] = None
            report_df["duplicate_score"] = None
            report_df["is_exact_match"] = False

            report_df["has_unresolved_issues"] = False

            # Add naming convention results if available
            if naming_results and "name" in report_df.columns:
                # Create a mapping from entity name to result
                naming_map = {result["entity"]: result for result in naming_results}

                # Update report DataFrame
                for idx, row in report_df.iterrows():
                    entity = row.get("name")
                    if entity in naming_map:
                        result = naming_map[entity]
                        report_df.at[idx, "has_naming_issues"] = not result["valid"]

                        if not result["valid"]:
                            report_df.at[idx, "naming_violations"] = "; ".join(
                                result["violations"]
                            )
                            report_df.at[idx, "naming_suggestion"] = result[
                                "suggestion"
                            ]
                            report_df.at[idx, "naming_status"] = result["status"]

                            # Check if issue is unresolved
                            if result["status"] in ["pending", "rejected"]:
                                report_df.at[idx, "has_unresolved_issues"] = True

            # Add duplicate detection results if available
            if duplicate_results and "name" in report_df.columns:
                # Update report DataFrame
                for idx, row in report_df.iterrows():
                    entity = row.get("name")
                    if entity in duplicate_results:
                        result = duplicate_results[entity]

                        # Determine if this is an issue (non-exact matches are issues)
                        is_issue = result.get(
                            "best_match"
                        ) is not None and not result.get("is_exact_match", False)
                        report_df.at[idx, "has_duplicate_issues"] = is_issue

                        report_df.at[idx, "duplicate_match"] = result.get("best_match")
                        report_df.at[idx, "duplicate_score"] = result.get(
                            "best_score", 0
                        )
                        report_df.at[idx, "is_exact_match"] = result.get(
                            "is_exact_match", False
                        )

                        # Check if issue is unresolved (needs user decision)
                        if is_issue:
                            is_resolved = False
                            if user_decisions and entity in user_decisions:
                                is_resolved = user_decisions[entity].get("action") in [
                                    "accept",
                                    "reject",
                                    "new",
                                ]

                            if not is_resolved:
                                report_df.at[idx, "has_unresolved_issues"] = True

            # Write report to file
            write_csv_file(report_df, file_path)

            # Log to audit log
            self.log_validation_action(
                "comprehensive_report",
                "multiple_entities",
                {
                    "total_entries": len(report_df),
                    "entries_with_naming_issues": int(
                        report_df["has_naming_issues"].sum()
                    ),
                    "entries_with_duplicate_issues": int(
                        report_df["has_duplicate_issues"].sum()
                    ),
                    "entries_with_unresolved_issues": int(
                        report_df["has_unresolved_issues"].sum()
                    ),
                },
            )

            logger.info(f"Generated comprehensive report: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return None

    def create_updated_compliant_file(
        self, input_df, naming_results=None, duplicate_results=None, user_decisions=None
    ):
        """
        Create an updated file that complies with all naming conventions.

        Args:
            input_df (pandas.DataFrame): Original input DataFrame.
            naming_results (List[Dict], optional): Naming convention results.
            duplicate_results (Dict, optional): Duplicate detection results.
            user_decisions (Dict, optional): User decisions on validation issues.

        Returns:
            Tuple[pandas.DataFrame, pandas.DataFrame]:
                (Updated compliant DataFrame, DataFrame of removed non-compliant entries)
        """
        # Start with a copy of the input DataFrame
        updated_df = input_df.copy()

        # Track entries to remove
        to_remove = pd.Series([False] * len(updated_df), index=updated_df.index)

        # Apply naming convention corrections if available
        if naming_results and "name" in updated_df.columns:
            # Create a mapping from entity name to result
            naming_map = {result["entity"]: result for result in naming_results}

            # Update DataFrame with corrections
            for idx, row in updated_df.iterrows():
                entity = row.get("name")
                if entity in naming_map:
                    result = naming_map[entity]

                    # Apply correction if accepted or resolved
                    if result["status"] in ["accepted", "modified"] or result["valid"]:
                        if result["suggestion"]:
                            updated_df.at[idx, "name"] = result["suggestion"]
                    # Mark for removal if invalid and not corrected
                    elif not result["valid"] and result["status"] in [
                        "pending",
                        "rejected",
                    ]:
                        to_remove[idx] = True

        # Apply duplicate resolutions if available
        if duplicate_results and user_decisions and "name" in updated_df.columns:
            for idx, row in updated_df.iterrows():
                entity = row.get("name")
                if entity in duplicate_results and entity in user_decisions:
                    result = duplicate_results[entity]
                    decision = user_decisions[entity]

                    action = decision.get("action")

                    # If accepting a match, replace with the matched entity
                    if action == "accept" and "selected_match" in decision:
                        updated_df.at[idx, "name"] = decision["selected_match"]
                    # If modifying, use the modified value
                    elif action == "modify" and "modified_value" in decision:
                        updated_df.at[idx, "name"] = decision["modified_value"]
                    # If rejecting and not a new entity, mark for removal
                    elif action == "reject" and not result.get("is_new_entity", True):
                        to_remove[idx] = True

        # Split into compliant and non-compliant DataFrames
        removed_df = updated_df[to_remove].copy()
        updated_df = updated_df[~to_remove].copy()

        return updated_df, removed_df

    def generate_final_reports(
        self,
        input_df,
        naming_results=None,
        duplicate_results=None,
        user_decisions=None,
        update_original=True,
    ):
        """
        Generate all final reports and updated files.

        Args:
            input_df (pandas.DataFrame): Original input DataFrame.
            naming_results (List[Dict], optional): Naming convention results.
            duplicate_results (Dict, optional): Duplicate detection results.
            user_decisions (Dict, optional): User decisions on validation issues.
            update_original (bool, optional): Whether to update the original file.
                Defaults to True.

        Returns:
            Dict: Paths to all generated files.
        """
        report_files = {}
        timestamp = self._create_timestamp()

        # Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report(
            input_df,
            naming_results,
            duplicate_results,
            user_decisions,
            f"comprehensive_report_{timestamp}.csv",
        )
        if comprehensive_report:
            report_files["comprehensive_report"] = comprehensive_report

        # Generate unresolved issues report
        unresolved_report = self.generate_unresolved_issues_report(
            input_df, f"unresolved_issues_{timestamp}.csv"
        )
        if unresolved_report:
            report_files["unresolved_issues"] = unresolved_report

        # Generate audit log
        audit_log = self.export_audit_log(f"audit_log_{timestamp}.csv")
        if audit_log:
            report_files["audit_log"] = audit_log

        # Update original file if requested
        if update_original:
            updated_df, removed_df = self.create_updated_compliant_file(
                input_df, naming_results, duplicate_results, user_decisions
            )

            # Save updated file
            updated_file = os.path.join(
                self.output_dir, f"updated_file_{timestamp}.csv"
            )
            write_csv_file(updated_df, updated_file)
            report_files["updated_file"] = updated_file

            # Save removed entries if any
            if not removed_df.empty:
                removed_file = os.path.join(
                    self.output_dir, f"removed_entries_{timestamp}.csv"
                )
                write_csv_file(removed_df, removed_file)
                report_files["removed_entries"] = removed_file

        return report_files

