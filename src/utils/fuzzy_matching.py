"""
fuzzy_matching.py

A comprehensive entity matching system that uses multiple techniques:
- Levenshtein Distance
- Token-Based Matching
- Phonetic Matching (Soundex/Metaphone)
- Machine Learning-Based Matching
"""

import os
import re
import pandas as pd
import numpy as np
import logging
import jellyfish
import difflib
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
from fuzzywuzzy import fuzz
from metaphone import doublemetaphone
from functools import lru_cache
from typing import List, Dict, Tuple, Any, Optional, Union, Set

# Configure logging
logger = logging.getLogger("fuzzy_matching")

# Try to import ML libraries, but handle gracefully if not available
try:
    import tensorflow as tf
    import tensorflow_hub as hub

    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logger.warning("TensorFlow/BERT not available - ML matching will be disabled")


class EntityMatcher:
    """Class for entity matching using multiple techniques."""

    # Default thresholds for different matching techniques
    DEFAULT_THRESHOLDS = {
        "levenshtein": 0.85,  # Higher is more similar
        "token": 0.80,  # Higher is more similar
        "soundex": 0.75,  # Higher is more similar
        "metaphone": 0.75,  # Higher is more similar
        "ml": 0.90,  # Higher is more similar
        "combined": 0.85,  # Higher is more similar
    }

    # Business designations to remove before matching
    BUSINESS_DESIGNATIONS = [
        "LLC",
        "Inc",
        "Inc.",
        "Incorporated",
        "Corp",
        "Corp.",
        "Corporation",
        "Ltd",
        "Ltd.",
        "Limited",
        "L.P.",
        "LP",
        "LLP",
        "PLLC",
        "P.C.",
        "L.L.C.",
        "L.L.P.",
        "P.L.L.C.",
    ]

    def __init__(self, thresholds=None, use_ml=False, ml_model_path=None):
        """
        Initialize the EntityMatcher with configurable thresholds.

        Args:
            thresholds (dict, optional): Custom thresholds for each matching technique.
                Defaults to DEFAULT_THRESHOLDS.
            use_ml (bool, optional): Whether to use machine learning-based matching.
                Defaults to False.
            ml_model_path (str, optional): Path to a pre-trained ML model.
                Defaults to None.
        """
        # Set thresholds, using defaults for any missing values
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)

        # Initialize ML model if requested and available
        self.use_ml = use_ml and BERT_AVAILABLE
        self.ml_model = None

        if self.use_ml:
            try:
                self._load_ml_model(ml_model_path)
            except Exception as e:
                logger.error(f"Failed to load ML model: {str(e)}")
                self.use_ml = False

        # TF-IDF vectorizer for token-based matching
        self.vectorizer = None

        # Initialize similarity cache
        self.similarity_cache = {}

    def _load_ml_model(self, model_path=None):
        """
        Load a pre-trained ML model for entity matching.

        Args:
            model_path (str, optional): Path to the model. If None, use a default
                Universal Sentence Encoder from TensorFlow Hub.
        """
        if not BERT_AVAILABLE:
            logger.error("TensorFlow not available - cannot load ML model")
            return

        try:
            if model_path:
                self.ml_model = tf.saved_model.load(model_path)
            else:
                # Use Universal Sentence Encoder as default
                model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
                self.ml_model = hub.load(model_url)
                logger.info("Loaded Universal Sentence Encoder for ML matching")
        except Exception as e:
            logger.error(f"Error loading ML model: {str(e)}")
            self.use_ml = False

    def set_threshold(self, technique, value):
        """
        Set the threshold for a specific matching technique.

        Args:
            technique (str): The matching technique ('levenshtein', 'token', etc.)
            value (float): Threshold value between 0 and 1.

        Raises:
            ValueError: If threshold is invalid.
        """
        if technique not in self.thresholds:
            raise ValueError(f"Unknown matching technique: {technique}")

        if not 0 <= value <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        self.thresholds[technique] = value
        logger.info(f"Set {technique} threshold to {value}")

    def get_thresholds(self):
        """Get the current thresholds for all matching techniques."""
        return self.thresholds.copy()

    @lru_cache(maxsize=10000)
    def preprocess_entity(self, entity):
        """
        Preprocess entity name for matching by normalizing, removing business
        designations, and special characters.

        Args:
            entity (str): The entity name to preprocess.

        Returns:
            str: The preprocessed entity name.
        """
        if not entity or not isinstance(entity, str):
            return ""

        # Convert to lowercase
        processed = entity.lower()

        # Remove business designations
        for designation in self.BUSINESS_DESIGNATIONS:
            pattern = r"\b" + re.escape(designation.lower()) + r"\b"
            processed = re.sub(pattern, "", processed)

        # Remove special characters except spaces
        processed = re.sub(r"[^\w\s]", "", processed)

        # Replace multiple spaces with a single space
        processed = re.sub(r"\s+", " ", processed).strip()

        return processed

    @lru_cache(maxsize=10000)
    def compute_similarity(self, str1, str2, method):
        """
        Compute similarity between two strings using specified method.

        Args:
            str1 (str): First string
            str2 (str): Second string
            method (str): Similarity method ('levenshtein', 'token', etc.)

        Returns:
            float: Similarity score between 0 and 1
        """
        # Early termination for identical strings
        if str1 == str2:
            return 1.0

        # Early termination for empty strings
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Pre-filter based on length difference
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))
        if max_len > 0 and len_diff / max_len > 0.5:
            return 0.0

        # Get cached result if available
        cache_key = (str1, str2, method)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Compute similarity based on method
        if method == "levenshtein":
            result = self.compute_levenshtein_similarity(str1, str2)
        elif method == "token":
            result = self.compute_token_similarity(str1, str2)
        elif method == "soundex":
            result = self.compute_soundex_similarity(str1, str2)
        elif method == "metaphone":
            result = self.compute_metaphone_similarity(str1, str2)
        elif method == "ml":
            result = self.compute_ml_similarity(str1, str2)
        else:  # combined
            result = self.compute_combined_similarity(str1, str2)

        # Cache and return result
        self.similarity_cache[cache_key] = result
        return result

    def compute_levenshtein_similarity(self, str1, str2):
        """
        Compute Levenshtein similarity between two strings.

        Args:
            str1 (str): First string.
            str2 (str): Second string.

        Returns:
            float: Similarity score between 0 and 1.
        """
        # Early termination checks
        if str1 == str2:
            return 1.0
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Preprocess strings
        str1_proc = self.preprocess_entity(str1)
        str2_proc = self.preprocess_entity(str2)

        # Early termination if preprocessed strings match
        if str1_proc == str2_proc:
            return 1.0

        # Skip if length difference is too great
        len_diff = abs(len(str1_proc) - len(str2_proc))
        max_len = max(len(str1_proc), len(str2_proc))
        if max_len > 0 and len_diff / max_len > 0.5:
            return 0.0

        # Calculate Levenshtein distance
        distance = Levenshtein.distance(str1_proc, str2_proc)

        # Convert to similarity score (between 0 and 1)
        max_len = max(len(str1_proc), len(str2_proc))
        if max_len == 0:  # Both empty after preprocessing
            return 1.0

        similarity = 1 - (distance / max_len)
        return similarity

    def compute_token_similarity(self, str1, str2, method="cosine"):
        """
        Compute token-based similarity between two strings.

        Args:
            str1 (str): First string.
            str2 (str): Second string.
            method (str, optional): Similarity method - 'cosine', 'jaccard', or 'dice'.
                Defaults to 'cosine'.

        Returns:
            float: Similarity score between 0 and 1.
        """
        # Early termination checks
        if str1 == str2:
            return 1.0
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Preprocess strings
        str1_proc = self.preprocess_entity(str1)
        str2_proc = self.preprocess_entity(str2)

        # Early termination if preprocessed strings match
        if str1_proc == str2_proc:
            return 1.0

        # Split into tokens
        tokens1 = set(str1_proc.split())
        tokens2 = set(str2_proc.split())

        # Early termination for empty token sets
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        if method == "jaccard":
            # Jaccard similarity: intersection / union
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))

            return intersection / union if union > 0 else 0.0

        elif method == "dice":
            # Dice coefficient: 2 * intersection / (size1 + size2)
            intersection = len(tokens1.intersection(tokens2))
            return (
                (2 * intersection) / (len(tokens1) + len(tokens2))
                if (len(tokens1) + len(tokens2)) > 0
                else 0.0
            )

        else:  # Default to cosine similarity with TF-IDF
            # Initialize vectorizer if not already done
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(lowercase=True)

            # Create a mini-corpus with the two strings
            corpus = [str1_proc, str2_proc]

            try:
                # Transform to TF-IDF vectors
                tfidf_matrix = self.vectorizer.fit_transform(corpus)

                # Calculate cosine similarity
                cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(cos_sim)  # Convert from numpy type
            except Exception as e:
                logger.warning(f"Error computing cosine similarity: {str(e)}")
                return 0.0

    def compute_soundex_similarity(self, str1, str2):
        """
        Compute Soundex similarity between two strings.

        Args:
            str1 (str): First string.
            str2 (str): Second string.

        Returns:
            float: Similarity score between 0 and 1.
        """
        # Early termination checks
        if str1 == str2:
            return 1.0
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Preprocess strings
        str1_proc = self.preprocess_entity(str1)
        str2_proc = self.preprocess_entity(str2)

        # Early termination if preprocessed strings match
        if str1_proc == str2_proc:
            return 1.0

        # Split into tokens
        tokens1 = str1_proc.split()
        tokens2 = str2_proc.split()

        # If no valid tokens, return 0
        if not tokens1 or not tokens2:
            return 0.0

        # Get Soundex codes for each token
        soundex1 = [jellyfish.soundex(token) for token in tokens1 if token]
        soundex2 = [jellyfish.soundex(token) for token in tokens2 if token]

        # If no valid Soundex codes, return 0
        if not soundex1 or not soundex2:
            return 0.0

        # Find best matches for each token
        matches = 0
        total = len(soundex1) + len(soundex2)

        # For each token in str1, find best match in str2
        for s1 in soundex1:
            best_match = max((1.0 if s1 == s2 else 0.0) for s2 in soundex2)
            matches += best_match

        # For each token in str2, find best match in str1
        for s2 in soundex2:
            best_match = max((1.0 if s2 == s1 else 0.0) for s1 in soundex1)
            matches += best_match

        # Average match score
        return matches / total if total > 0 else 0.0

    def compute_metaphone_similarity(self, str1, str2):
        """
        Compute Double Metaphone similarity between two strings.

        Args:
            str1 (str): First string.
            str2 (str): Second string.

        Returns:
            float: Similarity score between 0 and 1.
        """
        # Early termination checks
        if str1 == str2:
            return 1.0
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Preprocess strings
        str1_proc = self.preprocess_entity(str1)
        str2_proc = self.preprocess_entity(str2)

        # Early termination if preprocessed strings match
        if str1_proc == str2_proc:
            return 1.0

        # Split into tokens
        tokens1 = str1_proc.split()
        tokens2 = str2_proc.split()

        # If no valid tokens, return 0
        if not tokens1 or not tokens2:
            return 0.0

        # Get Metaphone codes for each token
        metaphones1 = [doublemetaphone(token) for token in tokens1 if token]
        metaphones2 = [doublemetaphone(token) for token in tokens2 if token]

        # If no valid Metaphone codes, return 0
        if not metaphones1 or not metaphones2:
            return 0.0

        # Find best matches for each token
        matches = 0
        total = len(metaphones1) + len(metaphones2)

        # For each token in str1, find best match in str2
        for m1 in metaphones1:
            best_match = 0.0
            for m2 in metaphones2:
                # Check if any of the metaphone codes match
                if (
                    m1[0]
                    and m2[0]
                    and (
                        m1[0] == m2[0]
                        or (m1[1] and m1[1] == m2[0])
                        or (m2[1] and m1[0] == m2[1])
                        or (m1[1] and m2[1] and m1[1] == m2[1])
                    )
                ):
                    best_match = 1.0
                    break
            matches += best_match

        # For each token in str2, find best match in str1
        for m2 in metaphones2:
            best_match = 0.0
            for m1 in metaphones1:
                # Check if any of the metaphone codes match
                if (
                    m1[0]
                    and m2[0]
                    and (
                        m1[0] == m2[0]
                        or (m1[1] and m1[1] == m2[0])
                        or (m2[1] and m1[0] == m2[1])
                        or (m1[1] and m2[1] and m1[1] == m2[1])
                    )
                ):
                    best_match = 1.0
                    break
            matches += best_match

        # Average match score
        return matches / total if total > 0 else 0.0

    def compute_ml_similarity(self, str1, str2):
        """
        Compute machine learning-based similarity between two strings.

        Args:
            str1 (str): First string.
            str2 (str): Second string.

        Returns:
            float: Similarity score between 0 and 1.
        """
        # Early termination checks
        if str1 == str2:
            return 1.0
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Check if ML model is available
        if not self.use_ml or not self.ml_model:
            logger.warning("ML model not available - falling back to Levenshtein")
            return self.compute_levenshtein_similarity(str1, str2)

        try:
            # Get embeddings
            embeddings = self.ml_model([str1, str2])

            # Calculate cosine similarity
            sim = np.inner(embeddings[0], embeddings[1])

            # Normalize to 0-1 range
            return float(sim)

        except Exception as e:
            logger.error(f"Error computing ML similarity: {str(e)}")
            # Fall back to Levenshtein
            return self.compute_levenshtein_similarity(str1, str2)

    def compute_combined_similarity(self, str1, str2):
        """
        Compute combined similarity using multiple techniques.

        Args:
            str1 (str): First string.
            str2 (str): Second string.

        Returns:
            float: Combined similarity score between 0 and 1.
        """
        # Early termination checks
        if str1 == str2:
            return 1.0
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Compute similarities using different methods
        levenshtein_sim = self.compute_levenshtein_similarity(str1, str2)

        # Early termination if Levenshtein is perfect
        if levenshtein_sim == 1.0:
            return 1.0

        # Skip other comparisons if Levenshtein is very low
        if levenshtein_sim < 0.3:
            return levenshtein_sim

        token_sim = self.compute_token_similarity(str1, str2)

        # Skip phonetic comparisons if both string and token similarities are low
        if levenshtein_sim < 0.4 and token_sim < 0.4:
            return max(levenshtein_sim, token_sim)

        soundex_sim = self.compute_soundex_similarity(str1, str2)
        metaphone_sim = self.compute_metaphone_similarity(str1, str2)

        # ML similarity if available and strings are potentially similar
        if self.use_ml and self.ml_model and (levenshtein_sim > 0.5 or token_sim > 0.5):
            ml_sim = self.compute_ml_similarity(str1, str2)
            # Weighted average of all techniques
            combined = (
                0.25 * levenshtein_sim
                + 0.25 * token_sim
                + 0.15 * soundex_sim
                + 0.15 * metaphone_sim
                + 0.2 * ml_sim
            )
        else:
            # Weighted average without ML
            combined = (
                0.35 * levenshtein_sim
                + 0.35 * token_sim
                + 0.15 * soundex_sim
                + 0.15 * metaphone_sim
            )

        return combined

    def find_matches(self, entity, candidates, match_type="combined", threshold=None):
        """
        Find matches for an entity among candidates.

        Args:
            entity (str): The entity to match.
            candidates (List[str]): List of candidate entities to match against.
            match_type (str, optional): Type of matching to use. Options are:
                'levenshtein', 'token', 'soundex', 'metaphone', 'ml', 'combined'.
                Defaults to 'combined'.
            threshold (float, optional): Similarity threshold. If None, uses the
                configured threshold for the selected match_type.

        Returns:
            List[Tuple[str, float]]: List of (candidate, score) tuples, sorted by
                descending score, where score >= threshold.
        """
        if not entity or not candidates:
            return []

        # Use configured threshold if not specified
        if threshold is None:
            threshold = self.thresholds.get(match_type, 0.85)

        # Get preprocessed entity
        entity_proc = self.preprocess_entity(entity)

        # For exact matches, we can just check equality
        exact_matches = [
            c for c in candidates if self.preprocess_entity(c) == entity_proc
        ]
        if exact_matches:
            return [(m, 1.0) for m in exact_matches]

        # Filter candidates by first letter to reduce comparisons
        if entity_proc:
            first_letter = entity_proc[0] if entity_proc else ""
            filtered_candidates = [
                c
                for c in candidates
                if self.preprocess_entity(c)
                and self.preprocess_entity(c)[0] == first_letter
            ]

            # If no filtered candidates, fall back to full list
            if not filtered_candidates:
                filtered_candidates = candidates
        else:
            filtered_candidates = candidates

        # Filter by length difference
        entity_len = len(entity_proc)
        len_filtered_candidates = [
            c
            for c in filtered_candidates
            if abs(len(self.preprocess_entity(c)) - entity_len) <= entity_len * 0.5
        ]

        # If no length-filtered candidates, fall back to all filtered candidates
        if not len_filtered_candidates:
            len_filtered_candidates = filtered_candidates

        # Calculate similarities
        matches = []
        for candidate in len_filtered_candidates:
            score = self.compute_similarity(entity, candidate, match_type)
            if score >= threshold:
                matches.append((candidate, score))

        # Sort by score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def find_all_matches(
        self, entities, candidates, match_type="combined", threshold=None
    ):
        """
        Find matches for multiple entities among candidates.

        Args:
            entities (List[str]): The entities to match.
            candidates (List[str]): List of candidate entities to match against.
            match_type (str, optional): Type of matching to use.
                Defaults to 'combined'.
            threshold (float, optional): Similarity threshold.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Dictionary mapping each entity to
                a list of (candidate, score) tuples, sorted by descending score.
        """
        results = {}

        # Preprocess all candidates once
        preprocessed_candidates = {c: self.preprocess_entity(c) for c in candidates}

        # Create an index of candidates by first letter
        candidates_by_letter = defaultdict(list)
        for c in candidates:
            proc_c = preprocessed_candidates[c]
            if proc_c:
                candidates_by_letter[proc_c[0]].append(c)

        for entity in entities:
            # Preprocess entity
            entity_proc = self.preprocess_entity(entity)

            # For exact matches
            exact_matches = [
                c for c in candidates if preprocessed_candidates[c] == entity_proc
            ]
            if exact_matches:
                results[entity] = [(m, 1.0) for m in exact_matches]
                continue

            # Filter candidates by first letter
            if entity_proc:
                first_letter = entity_proc[0]
                filtered_candidates = candidates_by_letter.get(first_letter, candidates)
            else:
                filtered_candidates = candidates

            # Find matches with filtered candidates
            matches = self.find_matches(
                entity, filtered_candidates, match_type, threshold
            )
            results[entity] = matches

        return results

    def detect_duplicates_within_list(
        self, entities, match_type="combined", threshold=None
    ):
        """
        Detect potential duplicates within a list of entities.

        Args:
            entities (List[str]): List of entities to check for duplicates.
            match_type (str, optional): Type of matching to use.
                Defaults to 'combined'.
            threshold (float, optional): Similarity threshold.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Dictionary mapping each entity to
                a list of potential duplicates with their similarity scores.
        """
        results = {}

        # Preprocess all entities once
        preprocessed_entities = {e: self.preprocess_entity(e) for e in entities}

        # Create an index of entities by first letter
        entities_by_letter = defaultdict(list)
        for e in entities:
            proc_e = preprocessed_entities[e]
            if proc_e:
                entities_by_letter[proc_e[0]].append(e)

        # Process each entity
        for i, entity in enumerate(entities):
            # Skip empty entities
            if not entity:
                continue

            # Get preprocessed entity
            entity_proc = preprocessed_entities[entity]

            # Find matches with other entities by first letter for efficiency
            if entity_proc:
                first_letter = entity_proc[0]
                candidates = entities_by_letter.get(first_letter, [])
                candidates = [c for c in candidates if c != entity]
            else:
                candidates = [e for e in entities if e != entity]

            # Find matches with filtered candidates
            matches = self.find_matches(entity, candidates, match_type, threshold)

            if matches:
                results[entity] = matches

        return results

    def match_with_authoritative_list(
        self, entities, auth_list, match_type="combined", threshold=None
    ):
        """
        Match entities against an authoritative list.

        Args:
            entities (List[str]): List of entities to match.
            auth_list (List[str]): Authoritative list of entities.
            match_type (str, optional): Type of matching to use.
                Defaults to 'combined'.
            threshold (float, optional): Similarity threshold.

        Returns:
            Dict[str, Dict]: Dictionary with detailed matching results:
                {
                    "entity_name": {
                        "matches": [("match1", score1), ("match2", score2), ...],
                        "best_match": "best_matching_entity",
                        "best_score": float,
                        "is_exact_match": bool,
                        "is_new_entity": bool
                    }
                }
        """
        results = {}

        # Preprocess all entities and auth_list once
        preprocessed_entities = {e: self.preprocess_entity(e) for e in entities}
        preprocessed_auth = {a: self.preprocess_entity(a) for a in auth_list}

        # Create an index of auth_list by first letter
        auth_by_letter = defaultdict(list)
        for a in auth_list:
            proc_a = preprocessed_auth[a]
            if proc_a:
                auth_by_letter[proc_a[0]].append(a)

        # Process each entity
        for entity in entities:
            # Skip empty entities
            if not entity:
                continue

            # Get preprocessed entity
            entity_proc = preprocessed_entities[entity]

            # Check for exact matches
            exact_matches = [
                a for a in auth_list if preprocessed_auth[a] == entity_proc
            ]

            if exact_matches:
                # Found exact match
                best_match = exact_matches[0]
                results[entity] = {
                    "matches": [(best_match, 1.0)],
                    "best_match": best_match,
                    "best_score": 1.0,
                    "is_exact_match": True,
                    "is_new_entity": False,
                }
                continue

            # Filter auth_list by first letter for efficiency
            if entity_proc:
                first_letter = entity_proc[0]
                filtered_auth = auth_by_letter.get(first_letter, auth_list)
            else:
                filtered_auth = auth_list

            # Find matches in filtered authoritative list
            matches = self.find_matches(entity, filtered_auth, match_type, threshold)

            # Determine if this is a new entity
            is_new_entity = True
            best_match = None
            best_score = 0.0

            if matches:
                best_match, best_score = matches[0]
                # Check if above automatic linking threshold
                if best_score >= self.thresholds.get("combined", 0.85):
                    is_new_entity = False

            # Store results
            results[entity] = {
                "matches": matches,
                "best_match": best_match,
                "best_score": best_score,
                "is_exact_match": False,
                "is_new_entity": is_new_entity,
            }

        return results

    def process_user_decisions(self, match_results, user_decisions):
        """
        Process user decisions on entity matches.

        Args:
            match_results (Dict): Results from match_with_authoritative_list.
            user_decisions (Dict): User decisions with structure:
                {
                    "entity_name": {
                        "action": "accept|reject|modify",
                        "selected_match": "selected_entity" (if accept),
                        "modified_value": "new_value" (if modify)
                    }
                }

        Returns:
            Dict: Updated match results with user decisions applied.
        """
        results = match_results.copy()

        for entity, decision in user_decisions.items():
            if entity not in results:
                logger.warning(f"Entity '{entity}' not found in match results")
                continue

            action = decision.get("action")

            if action == "accept":
                selected_match = decision.get("selected_match")
                if selected_match:
                    # Update with accepted match
                    results[entity]["best_match"] = selected_match
                    results[entity]["is_new_entity"] = False

                    # Find score for selected match
                    for match, score in results[entity]["matches"]:
                        if match == selected_match:
                            results[entity]["best_score"] = score
                            break

            elif action == "reject":
                # Mark as new entity
                results[entity]["best_match"] = None
                results[entity]["best_score"] = 0.0
                results[entity]["is_new_entity"] = True

            elif action == "modify":
                modified_value = decision.get("modified_value")
                if modified_value:
                    # Update with modified value
                    results[entity]["best_match"] = modified_value
                    results[entity]["is_new_entity"] = False
                    # Set a high score for the modified match
                    results[entity]["best_score"] = 1.0

        return results

    def update_authoritative_list(
        self, match_results, auth_list, new_entities_validation=None
    ):
        """
        Update authoritative list with new entities based on match results.

        Args:
            match_results (Dict): Results from process_user_decisions.
            auth_list (List[str]): Current authoritative list.
            new_entities_validation (callable, optional): Function to validate and format
                new entities before adding them to the authoritative list.

        Returns:
            Tuple[List[str], List[str]]: (Updated authoritative list, List of added entities)
        """
        updated_list = auth_list.copy()
        added_entities = []

        for entity, result in match_results.items():
            if result["is_new_entity"]:
                # Validate and format new entity if validation function provided
                if new_entities_validation and callable(new_entities_validation):
                    validated_entity = new_entities_validation(entity)
                    if validated_entity and validated_entity not in updated_list:
                        updated_list.append(validated_entity)
                        added_entities.append(validated_entity)
                else:
                    # Add as-is if no validation function
                    if entity not in updated_list:
                        updated_list.append(entity)
                        added_entities.append(entity)

        return updated_list, added_entities
