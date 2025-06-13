"""
duplicate_detection.py

This module performs duplicate detection on entity data by comparing entries
against the AuthoritativeEntityList.csv using fuzzy matching techniques.
"""

import os
import sys
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
import time
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from utils.fuzzy_matching import EntityMatcher
from utils.file_handling import read_csv_file, write_csv_file, standardize_columns
from validation.validation_rules import ValidationRules

# Configure logging
logger = logging.getLogger("duplicate_detection")


class DuplicateDetector:
    """
    Class for detecting and handling duplicate entities by comparing against
    an authoritative entity list.
    """

    def __init__(self, auth_list_path=None, fuzzy_threshold=None):
        """
        Initialize the DuplicateDetector.

        Args:
            auth_list_path (str, optional): Path to the authoritative entity list CSV.
                Defaults to None, which will look for it in the default location.
            fuzzy_threshold (float, optional): Threshold for fuzzy matching (0-1).
                Defaults to None, which uses the default threshold from EntityMatcher.
        """
        # Set default authoritative list path if not provided
        if auth_list_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.auth_list_path = os.path.join(
                base_dir, "data", "AuthoritativeEntityList.csv"
            )
        else:
            self.auth_list_path = auth_list_path

        # Initialize matching engine
        thresholds = {}
        if fuzzy_threshold is not None:
            thresholds = {
                "levenshtein": fuzzy_threshold,
                "token": fuzzy_threshold,
                "soundex": fuzzy_threshold,
                "metaphone": fuzzy_threshold,
                "ml": fuzzy_threshold,
                "combined": fuzzy_threshold,
            }

        self.matcher = EntityMatcher(thresholds=thresholds)

        # Initialize validation rules for new entities
        self.validator = ValidationRules()

        # Initialize authoritative entity list
        self.auth_entities = []
        self.auth_df = None
        self.auth_index = {}  # Index for faster lookup
        self._load_authoritative_list()

    def _load_authoritative_list(self):
        """
        Load the authoritative entity list from CSV file.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.auth_df = read_csv_file(self.auth_list_path)
            self.auth_df = standardize_columns(self.auth_df)

            # Extract entity names from all relevant columns
            entity_columns = [
                col
                for col in ["name", "facility", "owner", "operator"]
                if col in self.auth_df.columns
            ]

            self.auth_entities = []
            for col in entity_columns:
                entities = self.auth_df[col].dropna().tolist()
                self.auth_entities.extend(entities)

            # Remove duplicates and empty strings
            self.auth_entities = [e for e in set(self.auth_entities) if e]

            # Create index for faster lookup
            self._create_entity_index()

            logger.info(
                f"Loaded {len(self.auth_entities)} entities from authoritative list"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading authoritative list: {str(e)}")
            # Initialize with empty list if file not found
            self.auth_entities = []
            self.auth_df = pd.DataFrame()
            return False

    def _create_entity_index(self):
        """Create an index of entities for faster lookup."""
        self.auth_index = {}
        for entity in self.auth_entities:
            if not entity or not isinstance(entity, str):
                continue

            # Use first 3 characters as key (normalized to lowercase)
            if len(entity) >= 3:
                key = entity[:3].lower()
                if key not in self.auth_index:
                    self.auth_index[key] = []
                self.auth_index[key].append(entity)

            # Also index with first word
            words = entity.split()
            if words:
                first_word = words[0].lower()
                if first_word not in self.auth_index:
                    self.auth_index[first_word] = []
                if entity not in self.auth_index[first_word]:
                    self.auth_index[first_word].append(entity)

    def _get_candidate_matches(self, entity):
        """
        Get candidate matches from the authoritative list using the index.

        Args:
            entity (str): Entity to match.

        Returns:
            list: List of candidate entities for matching.
        """
        if not entity or not isinstance(entity, str):
            return []

        candidates = set()

        # Use first 3 characters as key
        if len(entity) >= 3:
            key = entity[:3].lower()
            if key in self.auth_index:
                candidates.update(self.auth_index[key])

        # Also check first word
        words = entity.split()
        if words:
            first_word = words[0].lower()
            if first_word in self.auth_index:
                candidates.update(self.auth_index[first_word])

        # If no candidates found, use full list (fallback)
        if not candidates:
            return self.auth_entities

        return list(candidates)

    def detect_duplicates(self, input_df, entity_columns=None, report_progress=None):
        """
        Detect duplicates by comparing input dataframe entities against the authoritative list.

        Args:
            input_df (pandas.DataFrame): DataFrame containing entity data.
            entity_columns (List[str], optional): Columns to check for entity names.
                Defaults to None, which uses ["name", "facility", "owner", "operator"].
            report_progress (callable, optional): Function to report progress (0-100).

        Returns:
            Tuple: (DataFrame with duplicate detection results, Dict of duplicate info)
                The DataFrame includes new columns:
                - 'duplicate_match': Name of matching entity in authoritative list
                - 'duplicate_score': Similarity score
                - 'is_duplicate': Boolean flag for duplicates
                - 'is_exact_match': Boolean flag for exact matches

            The Dict contains detailed match information for each entity.
        """
        start_time = time.time()

        # Use default entity columns if not specified
        if entity_columns is None:
            entity_columns = ["name", "facility", "owner", "operator"]

        # Filter to columns that exist in the DataFrame
        entity_columns = [col for col in entity_columns if col in input_df.columns]

        if not entity_columns:
            logger.warning("No entity columns found in input DataFrame")
            return input_df, {}

        # Create result containers (use dictionaries for better performance)
        result_dict = {
            "duplicate_match": [None] * len(input_df),
            "duplicate_score": [0.0] * len(input_df),
            "is_duplicate": [False] * len(input_df),
            "is_exact_match": [False] * len(input_df),
        }

        # Dictionary to store detailed match information
        match_info = {}

        # Check each row for duplicates
        total_rows = len(input_df)
        for i, (idx, row) in enumerate(input_df.iterrows()):
            # Report progress every 10 items or for every item if small dataset
            if report_progress and (i % 10 == 0 or total_rows < 100):
                progress = int(25 + (i / total_rows) * 75)
                report_progress(progress)

            row_matches = {}

            # Check each entity column
            for col in entity_columns:
                entity = row.get(col)

                # Skip empty values
                if pd.isna(entity) or entity == "":
                    continue

                # Get string representation
                entity_str = str(entity)

                # Get candidate matches (using index for efficiency)
                candidates = self._get_candidate_matches(entity_str)

                # Use the matcher to find matches among candidates
                match_result = self.matcher.match_with_authoritative_list(
                    [entity_str], candidates
                ).get(entity_str, {})

                # Store match information for this column
                row_matches[col] = match_result

                # Update result dict if this is the best match so far
                best_score = match_result.get("best_score", 0)
                if best_score > result_dict["duplicate_score"][i]:
                    result_dict["duplicate_match"][i] = match_result.get("best_match")
                    result_dict["duplicate_score"][i] = best_score
                    result_dict["is_duplicate"][i] = (
                        best_score >= self.matcher.thresholds.get("combined", 0.85)
                    )
                    result_dict["is_exact_match"][i] = match_result.get(
                        "is_exact_match", False
                    )

            # Store match info for this row
            match_info[idx] = row_matches

        # Create result DataFrame
        result_df = input_df.copy()
        for col, values in result_dict.items():
            result_df[col] = values

        end_time = time.time()
        logger.info(
            f"Duplicate detection completed in {end_time - start_time:.2f} seconds"
        )

        return result_df, match_info

    def process_user_decisions(self, input_df, match_info, user_decisions):
        """
        Process user decisions on duplicate matches.

        Args:
            input_df (pandas.DataFrame): Original input DataFrame.
            match_info (Dict): Match information from detect_duplicates.
            user_decisions (Dict): User decisions with structure:
                {
                    row_idx: {
                        'action': 'accept|reject|new',
                        'column': 'name|facility|owner|operator',
                        'match_entity': 'entity_name' (if accept),
                    }
                }

        Returns:
            Tuple: (Updated DataFrame, List of new entities to add to authoritative list)
        """
        result_df = input_df.copy()
        new_entities = []

        # Process each decision
        for row_idx, decision in user_decisions.items():
            # Convert index to correct type
            if isinstance(row_idx, str) and row_idx.isdigit():
                row_idx = int(row_idx)

            # Skip if row doesn't exist
            if row_idx not in result_df.index:
                logger.warning(f"Row index {row_idx} not found in DataFrame")
                continue

            action = decision.get("action")
            column = decision.get("column")

            # Skip if missing required information
            if not action or not column or column not in result_df.columns:
                logger.warning(f"Invalid decision for row {row_idx}: {decision}")
                continue

            # Process based on action
            if action == "accept":
                match_entity = decision.get("match_entity")
                if match_entity:
                    # Mark as duplicate with accepted match
                    result_df.at[row_idx, "duplicate_match"] = match_entity
                    result_df.at[row_idx, "is_duplicate"] = True
                    result_df.at[row_idx, "is_exact_match"] = (
                        False  # Not exact since user had to confirm
                    )

            elif action == "reject":
                # Clear duplicate flags
                result_df.at[row_idx, "duplicate_match"] = None
                result_df.at[row_idx, "duplicate_score"] = 0.0
                result_df.at[row_idx, "is_duplicate"] = False
                result_df.at[row_idx, "is_exact_match"] = False

            elif action == "new":
                # Get entity value
                entity = result_df.at[row_idx, column]

                # Skip if empty
                if pd.isna(entity) or entity == "":
                    continue

                # Validate entity name with naming convention rules
                validated_name, violations = self.validator.apply_all_rules(str(entity))

                # Add to new entities list if valid
                if validated_name and validated_name not in new_entities:
                    new_entities.append(validated_name)

                # Clear duplicate flags
                result_df.at[row_idx, "duplicate_match"] = None
                result_df.at[row_idx, "duplicate_score"] = 0.0
                result_df.at[row_idx, "is_duplicate"] = False
                result_df.at[row_idx, "is_exact_match"] = False

        return result_df, new_entities

    def update_authoritative_list(self, new_entities):
        """
        Update the authoritative entity list with new entities.

        Args:
            new_entities (List[str]): List of new entities to add.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not new_entities:
            return True

        try:
            # Load the current authoritative list
            current_auth_df = read_csv_file(self.auth_list_path)
            current_auth_df = standardize_columns(current_auth_df)

            # Determine which column to add to
            if "name" in current_auth_df.columns:
                target_column = "name"
            else:
                # Use the first available entity column
                available_columns = [
                    col
                    for col in ["facility", "owner", "operator"]
                    if col in current_auth_df.columns
                ]

                if not available_columns:
                    # Create a name column if no entity columns exist
                    current_auth_df["name"] = []
                    target_column = "name"
                else:
                    target_column = available_columns[0]

            # Create new rows for each entity
            new_rows = []
            for entity in new_entities:
                new_row = {col: None for col in current_auth_df.columns}
                new_row[target_column] = entity
                new_rows.append(new_row)

            # Append new rows
            new_df = pd.DataFrame(new_rows)
            updated_df = pd.concat([current_auth_df, new_df], ignore_index=True)

            # Write updated dataframe back to CSV
            success = write_csv_file(updated_df, self.auth_list_path)

            if success:
                # Reload authoritative list
                self._load_authoritative_list()
                logger.info(
                    f"Added {len(new_entities)} new entities to authoritative list"
                )
                return True
            else:
                logger.error("Failed to write updated authoritative list")
                return False

        except Exception as e:
            logger.error(f"Error updating authoritative list: {str(e)}")
            return False

    def _process_batch_internal_duplicates(self, batch_data):
        """
        Process a batch of data for internal duplicate detection.

        Args:
            batch_data (tuple): (batch_df, entity_columns)

        Returns:
            tuple: (results_dict, duplicate_groups)
        """
        batch_df, entity_columns = batch_data

        # Create a matcher instance for this process
        matcher = EntityMatcher(thresholds=self.matcher.thresholds)

        # Initialize result containers
        results_dict = {
            "internal_duplicate_group": [None] * len(batch_df),
            "internal_duplicate_match": [None] * len(batch_df),
            "internal_duplicate_score": [0.0] * len(batch_df),
            "is_internal_duplicate": [False] * len(batch_df),
        }

        duplicate_groups = {}
        group_counter = 0

        # Process each entity column
        for col in entity_columns:
            if col not in batch_df.columns:
                continue

            # Extract entities from this column
            entities = batch_df[col].dropna().astype(str).tolist()

            # Create a mapping of entity to row indices
            entity_to_indices = {}
            for idx, row in batch_df.iterrows():
                entity = row.get(col)
                if pd.notna(entity) and entity != "":
                    entity_str = str(entity)
                    if entity_str not in entity_to_indices:
                        entity_to_indices[entity_str] = []
                    entity_to_indices[entity_str].append(idx)

            # Detect duplicates using optimized batch processing
            duplicates = matcher.detect_duplicates_within_list(entities)

            # Process each potential duplicate
            for entity, matches in duplicates.items():
                # Skip if no matches found or no row indices for this entity
                if not matches or entity not in entity_to_indices:
                    continue

                # Create a new duplicate group
                group_id = f"group_{group_counter}"
                group_counter += 1

                # Add original entity's indices to group
                group_members = entity_to_indices[entity].copy()

                # Add matching entities' indices to group
                for match, score in matches:
                    if match in entity_to_indices:
                        for idx in entity_to_indices[match]:
                            if idx not in group_members:
                                group_members.append(idx)

                # Update results with group information
                for idx in group_members:
                    i = batch_df.index.get_loc(idx)

                    # Skip if already in a group with higher score
                    if (
                        results_dict["is_internal_duplicate"][i]
                        and results_dict["internal_duplicate_score"][i] >= matches[0][1]
                    ):
                        continue

                    results_dict["internal_duplicate_group"][i] = group_id
                    results_dict["is_internal_duplicate"][i] = True

                    # First match is always the best match
                    if matches:
                        best_match, best_score = matches[0]
                        results_dict["internal_duplicate_match"][i] = best_match
                        results_dict["internal_duplicate_score"][i] = best_score

                # Store group information
                duplicate_groups[group_id] = {
                    "members": group_members,
                    "entities": [
                        batch_df.at[idx, col]
                        for idx in group_members
                        if idx in batch_df.index and pd.notna(batch_df.at[idx, col])
                    ],
                    "column": col,
                }

        return results_dict, duplicate_groups

    def detect_internal_duplicates(
        self, input_df, entity_columns=None, report_progress=None
    ):
        """
        Detect duplicates within the input DataFrame itself.

        Args:
            input_df (pandas.DataFrame): DataFrame to check for internal duplicates.
            entity_columns (List[str], optional): Columns to check for entity names.
            report_progress (callable, optional): Function to report progress (0-100).

        Returns:
            Tuple: (DataFrame with duplicate detection results, Dict of duplicate groups)
                The DataFrame includes new columns:
                - 'internal_duplicate_group': ID for duplicate group
                - 'internal_duplicate_match': Best matching entity within the dataset
                - 'internal_duplicate_score': Similarity score
                - 'is_internal_duplicate': Boolean flag for internal duplicates
        """
        start_time = time.time()

        # Use default entity columns if not specified
        if entity_columns is None:
            entity_columns = ["name", "facility", "owner", "operator"]

        # Filter to columns that exist in the DataFrame
        entity_columns = [col for col in entity_columns if col in input_df.columns]

        if not entity_columns:
            logger.warning("No entity columns found in input DataFrame")
            return input_df, {}

        # Report initial progress
        if report_progress:
            report_progress(5)

        # For small datasets, process directly
        if len(input_df) < 1000:
            batch_data = (input_df, entity_columns)
            results_dict, duplicate_groups = self._process_batch_internal_duplicates(
                batch_data
            )

            # Create result DataFrame
            result_df = input_df.copy()
            for col, values in results_dict.items():
                result_df[col] = values

            # Update progress
            if report_progress:
                report_progress(100)

            end_time = time.time()
            logger.info(
                f"Internal duplicate detection completed in {end_time - start_time:.2f} seconds"
            )

            return result_df, duplicate_groups

        # For larger datasets, use parallel processing
        # Determine optimal batch size and worker count
        cpu_count = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
        batch_size = max(100, len(input_df) // (cpu_count * 2))

        # Split DataFrame into batches
        df_batches = []
        for i in range(0, len(input_df), batch_size):
            end_idx = min(i + batch_size, len(input_df))
            df_batches.append(input_df.iloc[i:end_idx])

        # Prepare batch data
        batch_data = [(df_batch, entity_columns) for df_batch in df_batches]

        # Update progress
        if report_progress:
            report_progress(10)

        # Process batches in parallel
        all_results = []
        all_groups = {}

        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            for i, (results_dict, duplicate_groups) in enumerate(
                executor.map(self._process_batch_internal_duplicates, batch_data)
            ):
                all_results.append(results_dict)

                # Adjust group IDs to prevent conflicts
                prefix = f"batch_{i}_"
                renamed_groups = {}

                for group_id, group_info in duplicate_groups.items():
                    new_group_id = prefix + group_id
                    renamed_groups[new_group_id] = group_info

                    # Update group IDs in results
                    for j, group in enumerate(results_dict["internal_duplicate_group"]):
                        if group == group_id:
                            results_dict["internal_duplicate_group"][j] = new_group_id

                all_groups.update(renamed_groups)

                # Update progress
                if report_progress:
                    progress = 10 + int(90 * (i + 1) / len(batch_data))
                    report_progress(progress)

        # Combine results
        combined_results = {
            "internal_duplicate_group": [None] * len(input_df),
            "internal_duplicate_match": [None] * len(input_df),
            "internal_duplicate_score": [0.0] * len(input_df),
            "is_internal_duplicate": [False] * len(input_df),
        }

        start_idx = 0
        for results_dict in all_results:
            batch_size = len(results_dict["internal_duplicate_group"])

            for col in combined_results:
                combined_results[col][start_idx : start_idx + batch_size] = (
                    results_dict[col]
                )

            start_idx += batch_size

        # Create result DataFrame
        result_df = input_df.copy()
        for col, values in combined_results.items():
            result_df[col] = values

        end_time = time.time()
        logger.info(
            f"Internal duplicate detection completed in {end_time - start_time:.2f} seconds"
        )

        return result_df, all_groups

    def suggest_internal_duplicate_resolution(self, duplicate_groups, input_df):
        """
        Suggest resolutions for internal duplicates.

        Args:
            duplicate_groups (Dict): Duplicate groups from detect_internal_duplicates.
            input_df (pandas.DataFrame): Original input DataFrame.

        Returns:
            Dict: Suggested resolutions for each duplicate group.
        """
        suggestions = {}

        for group_id, group_info in duplicate_groups.items():
            entities = group_info["entities"]
            if not entities:
                continue

            # Find the best entity by:
            # 1. Applying naming conventions to all entities
            # 2. Selecting the one with fewest validation issues
            best_entity = None
            best_violations_count = float("inf")

            for entity in entities:
                if not entity or pd.isna(entity):
                    continue

                # Validate entity
                validated_name, violations = self.validator.apply_all_rules(str(entity))

                # Check if this is better than current best
                if len(violations) < best_violations_count:
                    best_entity = validated_name or str(entity)
                    best_violations_count = len(violations)

            # Store suggestion
            if best_entity:
                suggestions[group_id] = {
                    "keep_entity": best_entity,
                    "duplicate_entities": [
                        e for e in entities if str(e) != best_entity
                    ],
                }

        return suggestions

    def handle_internal_duplicates(self, input_df, duplicate_groups, user_decisions):
        """
        Handle internal duplicates based on user decisions.

        Args:
            input_df (pandas.DataFrame): Input DataFrame with duplicate detection results.
            duplicate_groups (Dict): Duplicate groups from detect_internal_duplicates.
            user_decisions (Dict): User decisions with structure:
                {
                    group_id: {
                        'action': 'keep|merge|custom',
                        'keep_entity': 'entity_to_keep' (if keep),
                        'custom_value': 'custom_value' (if custom)
                    }
                }

        Returns:
            pandas.DataFrame: Updated DataFrame with duplicates resolved.
        """
        result_df = input_df.copy()

        # Process each decision
        for group_id, decision in user_decisions.items():
            # Skip if group doesn't exist
            if group_id not in duplicate_groups:
                logger.warning(f"Group ID {group_id} not found in duplicate groups")
                continue

            action = decision.get("action")
            group_info = duplicate_groups[group_id]
            column = group_info.get("column")

            # Skip if missing required information
            if not action or not column or column not in result_df.columns:
                logger.warning(f"Invalid decision for group {group_id}: {decision}")
                continue

            # Get member indices for this group
            member_indices = group_info.get("members", [])

            # Process based on action
            if action == "keep":
                keep_entity = decision.get("keep_entity")
                if keep_entity:
                    # Update all members with the entity to keep
                    for idx in member_indices:
                        if idx in result_df.index:
                            result_df.at[idx, column] = keep_entity

            elif action == "merge":
                # Apply validation rules to all entities and keep the best one
                entities = [
                    result_df.at[idx, column]
                    for idx in member_indices
                    if idx in result_df.index and pd.notna(result_df.at[idx, column])
                ]

                best_entity = None
                best_violations_count = float("inf")

                for entity in entities:
                    if not entity or pd.isna(entity):
                        continue

                    # Validate entity
                    validated_name, violations = self.validator.apply_all_rules(
                        str(entity)
                    )

                    # Check if this is better than current best
                    if len(violations) < best_violations_count:
                        best_entity = validated_name or str(entity)
                        best_violations_count = len(violations)

                # Update all members with the best entity
                if best_entity:
                    for idx in member_indices:
                        if idx in result_df.index:
                            result_df.at[idx, column] = best_entity

            elif action == "custom":
                custom_value = decision.get("custom_value")
                if custom_value:
                    # Validate custom value
                    validated_name, violations = self.validator.apply_all_rules(
                        str(custom_value)
                    )
                    final_value = validated_name or custom_value

                    # Update all members with the custom value
                    for idx in member_indices:
                        if idx in result_df.index:
                            result_df.at[idx, column] = final_value

        return result_df
