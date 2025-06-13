"""
naming_convention.py

This module checks entity names against predefined validation rules
and returns detected violations with suggested corrections.
"""

import os
import pandas as pd
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import validation rules
from validation.validation_rules import ValidationRules
from src.utils.file_handling import log_validation_error

# Configure logging
logger = logging.getLogger("naming_convention")


class NamingConventionValidator:
    """Class for validating entity names against naming conventions."""

    def __init__(self, fuzzy_threshold=None):
        """
        Initialize the validator with validation rules.

        Args:
            fuzzy_threshold (int, optional): Threshold for fuzzy matching (0-100).
        """
        self.validator = ValidationRules(fuzzy_threshold)

    def validate_entity(self, entity_name: str) -> Dict[str, Any]:
        """
        Validate a single entity name against naming convention rules.

        Args:
            entity_name (str): The entity name to validate.

        Returns:
            Dict: Dictionary containing violation information:
                - 'entity': Original entity name
                - 'valid': Boolean indicating if valid
                - 'violations': List of rule violations
                - 'suggestion': Suggested correction
                - 'status': Current status ('pending', 'accepted', 'rejected', 'modified')
        """
        if not entity_name or not isinstance(entity_name, str):
            return {
                "entity": str(entity_name),
                "valid": False,
                "violations": ["Empty or invalid entity name"],
                "suggestion": "",
                "status": "pending",
            }

        # Apply validation rules to get corrected name and list of violations
        corrected_name, violations = self.validator.apply_all_rules(entity_name)

        # Create result dictionary
        result = {
            "entity": entity_name,
            "valid": len(violations) == 0,
            "violations": violations,
            "suggestion": corrected_name if violations else "",
            "status": "pending" if violations else "valid",
        }

        # Log any validation errors
        if violations:
            for violation in violations:
                log_validation_error(entity_name, "name", violation)

        return result

    def validate_entities(self, entities: List[str]) -> List[Dict[str, Any]]:
        """
        Validate multiple entity names against naming convention rules.

        Args:
            entities (List[str]): List of entity names to validate.

        Returns:
            List[Dict]: List of dictionaries containing violation information.
        """
        results = []
        for entity in entities:
            result = self.validate_entity(entity)
            results.append(result)

        return results

    def validate_dataframe(
        self, df, entity_column="name"
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Validate entity names in a DataFrame.

        Args:
            df: Pandas DataFrame containing entity data.
            entity_column (str, optional): Column name containing entity names.
                Defaults to 'name'.

        Returns:
            Tuple: (List of validation results, DataFrame with added validation columns)
        """
        if entity_column not in df.columns:
            logger.error(f"Column '{entity_column}' not found in DataFrame")
            return [], df

        validation_results = []

        # Add validation columns to DataFrame
        df["validation_valid"] = True
        df["validation_violations"] = None
        df["validation_suggestion"] = None
        df["validation_status"] = "valid"

        # Validate each row
        for idx, row in df.iterrows():
            entity_name = row[entity_column]
            if pd.isna(entity_name):  # Skip NaN values
                continue

            result = self.validate_entity(str(entity_name))
            validation_results.append(result)

            # Update DataFrame with validation results
            df.at[idx, "validation_valid"] = result["valid"]
            df.at[idx, "validation_violations"] = (
                "; ".join(result["violations"]) if result["violations"] else None
            )
            df.at[idx, "validation_suggestion"] = result["suggestion"]
            df.at[idx, "validation_status"] = result["status"]

        return validation_results, df

    def process_user_decisions(
        self,
        validation_results: List[Dict[str, Any]],
        user_decisions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process user decisions on validation suggestions.

        Args:
            validation_results (List[Dict]): Original validation results.
            user_decisions (List[Dict]): User decisions with the following structure:
                [
                    {
                        'entity': 'Original Entity Name',
                        'action': 'accept|reject|modify',
                        'modified_value': 'New value if modified'
                    },
                    ...
                ]

        Returns:
            List[Dict]: Updated validation results with user decisions applied.
        """
        # Create a mapping of entity names to results for easy lookup
        results_map = {result["entity"]: result for result in validation_results}

        # Process each user decision
        for decision in user_decisions:
            entity = decision.get("entity")
            action = decision.get("action")

            if entity not in results_map:
                logger.warning(f"Entity '{entity}' not found in validation results")
                continue

            result = results_map[entity]

            if action == "accept":
                result["status"] = "accepted"
                # Entity name remains the suggested correction

            elif action == "reject":
                result["status"] = "rejected"
                # Entity name remains unchanged

            elif action == "modify":
                modified_value = decision.get("modified_value", "")
                if modified_value:
                    result["suggestion"] = modified_value
                    result["status"] = "modified"
                else:
                    logger.warning(f"No modified value provided for entity '{entity}'")

        return list(results_map.values())

    def apply_corrections(self, df, entity_column="name", only_accepted=True):
        """
        Apply corrections to a DataFrame based on validation status.

        Args:
            df: Pandas DataFrame with validation columns.
            entity_column (str, optional): Column name containing entity names.
                Defaults to 'name'.
            only_accepted (bool, optional): If True, only apply corrections for
                'accepted' or 'modified' statuses. If False, apply all suggestions
                except 'rejected'. Defaults to True.

        Returns:
            DataFrame: Updated DataFrame with corrections applied.
        """
        if not all(
            col in df.columns for col in ["validation_suggestion", "validation_status"]
        ):
            logger.error("DataFrame missing required validation columns")
            return df

        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Apply corrections based on status
        for idx, row in result_df.iterrows():
            status = row["validation_status"]
            suggestion = row["validation_suggestion"]

            if pd.isna(suggestion) or suggestion == "":
                continue

            if only_accepted and status in ["accepted", "modified"]:
                result_df.at[idx, entity_column] = suggestion
            elif not only_accepted and status != "rejected":
                result_df.at[idx, entity_column] = suggestion

        return result_df

