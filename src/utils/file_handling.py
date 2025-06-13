"""
file_handling.py

This module contains utilities for handling CSV files, including reading, writing,
and validating entity data.
"""

import os
import pandas as pd
import logging
import csv
from datetime import datetime
from difflib import SequenceMatcher
import re

# Configure logging
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
)
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir, f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("file_handler")


def read_csv_file(file_path, required_columns=None):
    """
    Read a CSV file into a pandas DataFrame and validate it has required columns.

    Args:
        file_path (str): Path to the CSV file.
        required_columns (list, optional): List of column names that should be present.
            Defaults to None. If None, at least one of ["name", "facility", "owner", "operator"]
            must be present.

    Returns:
        pandas.DataFrame: The DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is incorrect or required columns are missing.
    """
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # Try to read the CSV file
        df = pd.read_csv(file_path)

        # Check if DataFrame is empty
        if df.empty:
            error_msg = f"Empty CSV file: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate required columns
        if required_columns is None:
            # Default required columns - at least one must be present
            default_required = ["name", "facility", "owner", "operator"]
            if not any(col in df.columns for col in default_required):
                error_msg = f"CSV is missing at least one required column from {default_required}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            # Check specific required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"CSV is missing required columns: {missing_columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        logger.info(f"Successfully read CSV file: {file_path}")
        return df

    except pd.errors.ParserError:
        error_msg = f"Error parsing CSV file: {file_path}. Invalid format."
        logger.error(error_msg)
        raise ValueError(error_msg)


def write_csv_file(df, output_path, index=False):
    """
    Write a DataFrame to a CSV file.

    Args:
        df (pandas.DataFrame): The DataFrame to write.
        output_path (str): Path to save the CSV file.
        index (bool, optional): Whether to include the index. Defaults to False.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write to CSV
        df.to_csv(output_path, index=index)
        logger.info(f"Successfully wrote CSV file: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error writing CSV file {output_path}: {str(e)}")
        return False


def log_validation_error(entity_name, field_name, error_message):
    """
    Log a validation error.

    Args:
        entity_name (str): The name of the entity with the error.
        field_name (str): The field containing the error.
        error_message (str): The error message describing the issue.
    """
    logger.error(
        f"Validation error in {field_name} for entity '{entity_name}': {error_message}"
    )


def standardize_columns(df):
    """
    Standardize column names by converting to lowercase,
    removing spaces, and other standardization tasks.

    Args:
        df (pandas.DataFrame): The DataFrame to standardize.

    Returns:
        pandas.DataFrame: The standardized DataFrame.
    """
    # Convert all column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Replace spaces with underscores in column names
    df.columns = [col.replace(" ", "_") for col in df.columns]

    # Map common variations of entity columns to standard names
    column_mappings = {
        "entity_name": "name",
        "entity": "name",
        "company_name": "name",
        "company": "name",
        "business_name": "name",
        "business": "name",
        "facility_name": "facility",
        "operator_name": "operator",
        "owner_name": "owner",
    }

    # Rename columns based on mappings
    df = df.rename(columns=lambda x: column_mappings.get(x, x))

    # Convert blank/None values to NaN for consistency
    df = df.replace("", pd.NA)

    logger.info("Standardized DataFrame columns")
    return df


def detect_internal_duplicates(df, threshold=0.85):
    """
    Detect potential duplicate entities within the input DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing entity data.
        threshold (float, optional): Similarity threshold (0-1) for fuzzy matching.
            Defaults to 0.85.

    Returns:
        pandas.DataFrame: DataFrame with added columns:
            - 'duplicate_group': Assigns a group ID to each set of duplicates
            - 'is_duplicate': Boolean indicating if the entity is a duplicate
            - 'best_match': The name of the best matching entity (if any)
            - 'match_score': The similarity score with the best match
    """
    # Create columns for tracking duplicates
    df["duplicate_group"] = None
    df["is_duplicate"] = False
    df["best_match"] = None
    df["match_score"] = 0.0

    # Get entity names and normalize for comparison
    entity_columns = [
        col for col in ["name", "facility", "owner", "operator"] if col in df.columns
    ]

    if not entity_columns:
        logger.warning("No entity columns found for duplicate detection")
        return df

    # Process each entity column
    for column in entity_columns:
        # Skip if column has no values
        if df[column].isna().all():
            continue

        # Clean strings for comparison
        df[f"clean_{column}"] = (
            df[column].fillna("").astype(str).apply(clean_string_for_comparison)
        )

        # Dictionary to store duplicate groups
        duplicate_groups = {}
        group_counter = 0

        # Compare each pair of entities
        values = df[f"clean_{column}"].tolist()
        indices = df.index.tolist()

        for i, (idx1, val1) in enumerate(zip(indices, values)):
            # Skip if already assigned to a duplicate group
            if df.at[idx1, "duplicate_group"] is not None:
                continue

            # Find potential duplicates
            group_members = []

            for j, (idx2, val2) in enumerate(zip(indices, values)):
                if i == j:  # Don't compare with self
                    continue

                # Calculate similarity
                similarity = string_similarity(val1, val2)

                # If similar enough, add to group
                if similarity >= threshold:
                    # Update best match if this is better than previous
                    if similarity > df.at[idx2, "match_score"]:
                        df.at[idx2, "best_match"] = df.at[idx1, column]
                        df.at[idx2, "match_score"] = similarity

                    group_members.append(idx2)

            # If duplicates found, create a group
            if group_members:
                group_id = f"group_{group_counter}"
                group_counter += 1

                # Assign self to group
                df.at[idx1, "duplicate_group"] = group_id

                # Assign all members to group
                for member_idx in group_members:
                    df.at[member_idx, "duplicate_group"] = group_id
                    df.at[member_idx, "is_duplicate"] = True

        # Drop temporary cleaning column
        df = df.drop(f"clean_{column}", axis=1)

    logger.info(f"Detected {df['is_duplicate'].sum()} potential internal duplicates")
    return df


def clean_string_for_comparison(text):
    """
    Clean a string for comparison in duplicate detection.
    Removes special characters, extra whitespace, and converts to lowercase.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove business designations
    business_designations = [
        "llc",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "ltd",
        "limited",
        "lp",
        "llp",
        "pllc",
        "pc",
    ]

    for designation in business_designations:
        text = re.sub(r"\b" + designation + r"\b", "", text)

    # Standardize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def string_similarity(str1, str2):
    """
    Calculate the similarity between two strings using SequenceMatcher.

    Args:
        str1 (str): First string.
        str2 (str): Second string.

    Returns:
        float: Similarity score between 0 and 1.
    """
    if not str1 and not str2:  # Both empty
        return 1.0
    if not str1 or not str2:  # One empty
        return 0.0

    return SequenceMatcher(None, str1, str2).ratio()


def compare_with_authoritative_list(input_df, auth_file_path, threshold=0.85):
    """
    Compare entities in the input DataFrame with the authoritative entity list.

    Args:
        input_df (pandas.DataFrame): DataFrame containing input entities.
        auth_file_path (str): Path to the authoritative entity list CSV.
        threshold (float, optional): Similarity threshold (0-1) for fuzzy matching.
            Defaults to 0.85.

    Returns:
        pandas.DataFrame: Input DataFrame with added columns:
            - 'auth_match': The name of the matching entity in the authoritative list
            - 'auth_match_score': The similarity score with the authoritative match
            - 'is_new_entity': Boolean indicating if this appears to be a new entity
    """
    # Read authoritative list
    try:
        auth_df = read_csv_file(auth_file_path)
    except Exception as e:
        logger.error(f"Error loading authoritative list: {str(e)}")
        # Add placeholder columns
        input_df["auth_match"] = None
        input_df["auth_match_score"] = 0.0
        input_df["is_new_entity"] = True
        return input_df

    # Standardize authoritative list columns
    auth_df = standardize_columns(auth_df)

    # Add columns for tracking matches
    input_df["auth_match"] = None
    input_df["auth_match_score"] = 0.0
    input_df["is_new_entity"] = True

    # Get entity names from authoritative list
    auth_entity_columns = [
        col
        for col in ["name", "facility", "owner", "operator"]
        if col in auth_df.columns
    ]

    if not auth_entity_columns:
        logger.warning("No entity columns found in authoritative list")
        return input_df

    # Create cleaned versions of authoritative entities for comparison
    auth_entities = {}
    for column in auth_entity_columns:
        auth_entities[column] = []
        for idx, value in auth_df[column].items():
            if pd.notna(value):
                clean_value = clean_string_for_comparison(str(value))
                auth_entities[column].append((idx, value, clean_value))

    # Get entity columns in input data
    input_entity_columns = [
        col
        for col in ["name", "facility", "owner", "operator"]
        if col in input_df.columns
    ]

    # Compare each input entity against authoritative list
    for idx, row in input_df.iterrows():
        best_match = None
        best_score = 0

        for input_col in input_entity_columns:
            # Skip if input value is missing
            if pd.isna(row[input_col]):
                continue

            input_value = str(row[input_col])
            clean_input = clean_string_for_comparison(input_value)

            for auth_col in auth_entity_columns:
                for auth_idx, auth_orig, auth_clean in auth_entities[auth_col]:
                    # Calculate similarity
                    similarity = string_similarity(clean_input, auth_clean)

                    # If perfect match (100%)
                    if similarity == 1.0:
                        input_df.at[idx, "auth_match"] = auth_orig
                        input_df.at[idx, "auth_match_score"] = 1.0
                        input_df.at[idx, "is_new_entity"] = False
                        best_match = auth_orig
                        best_score = 1.0
                        break

                    # If better than current best and above threshold
                    elif similarity > best_score and similarity >= threshold:
                        best_match = auth_orig
                        best_score = similarity

                # If perfect match found, break out of auth column loop
                if best_score == 1.0:
                    break

            # If perfect match found, break out of input column loop
            if best_score == 1.0:
                break

        # Update with best match if found
        if best_match and best_score > input_df.at[idx, "auth_match_score"]:
            input_df.at[idx, "auth_match"] = best_match
            input_df.at[idx, "auth_match_score"] = best_score
            # If score is above threshold, mark as not a new entity
            if best_score >= threshold:
                input_df.at[idx, "is_new_entity"] = False

    logger.info(
        f"Found {input_df['is_new_entity'].value_counts().get(False, 0)} matches with authoritative list"
    )
    return input_df



