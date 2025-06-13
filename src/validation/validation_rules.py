"""
validation_rules.py

This module contains the rules for entity name validation according to the
specified naming convention standards.
"""

import re
import string


class ValidationRules:
    """Class containing validation rules and methods for entity names."""

    # Default configuration values
    DEFAULT_FUZZY_MATCH_THRESHOLD = 85

    # Business designations to remove
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

    # Common abbreviations to expand
    ABBREVIATIONS = {
        "Co": "Company",
        "CO": "Company",
        "Co.": "Company",
        "Bros": "Brothers",
        "BROS": "Brothers",
        "Bros.": "Brothers",
        "Mfg": "Manufacturing",
        "MFG": "Manufacturing",
        "Mfg.": "Manufacturing",
        "Intl": "International",
        "INTL": "International",
        "Intl.": "International",
        "&": "and",
        "CORP": "Corporation",
        "Corp": "Corporation",
        "COOP": "Cooperative",
        "Coop": "Cooperative",
        "ELEC": "Electric",
        "Elec": "Electric",
        "ASSN": "Association",
        "Assn": "Association",
        "INC": "Incorporated",
        "Inc": "Incorporated",
    }

    # Government entity patterns
    GOVT_ENTITY_PATTERNS = {
        "state": r"(?i)^State of ([A-Za-z ]+)$",
        "county": r"(?i)^County of ([A-Za-z ]+)(\s*[-]?\s*\([A-Z]{2}\))?$",
        "city": r"(?i)^(City|Town|Village|Borough) of ([A-Za-z ]+)(\s*[-]?\s*\([A-Z]{2}\))?$",
    }

    # US state abbreviations
    US_STATES = {
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
        "DC",
        "AS",
        "GU",
        "MP",
        "PR",
        "VI",
    }

    # Common country and organization acronyms we may want to allow
    COMMON_COUNTRY_ACRONYMS = {"USA", "UK", "EU", "UN", "US"}

    def __init__(self, fuzzy_threshold=None):
        """
        Initialize ValidationRules with configurable settings.

        Args:
            fuzzy_threshold (int, optional): Threshold for fuzzy matching (0-100).
                Defaults to DEFAULT_FUZZY_MATCH_THRESHOLD.
        """
        self.fuzzy_threshold = fuzzy_threshold or self.DEFAULT_FUZZY_MATCH_THRESHOLD

    def apply_all_rules(self, entity_name):
        """
        Apply all validation rules to an entity name.

        Args:
            entity_name (str): The entity name to validate.

        Returns:
            tuple: (validated_name, list of violations)
        """
        if not entity_name or not isinstance(entity_name, str):
            return "", ["Empty or invalid entity name"]

        violations = []

        # Track the original and current state for reporting
        original_name = entity_name
        current_name = entity_name

        # Rule 1: Check if it appears to be an acronym
        if self._is_likely_acronym(current_name):
            violations.append(
                "Entity appears to be an acronym - full name should be spelled out"
            )
            # Don't suggest a correction for acronyms since we don't know the expanded form
            # Just keep track of the violation

        # Rule 2: Replace ampersands
        if "&" in current_name:
            violations.append("Ampersand used instead of 'and'")
            current_name = self._replace_ampersands(current_name)

        # Rule 3: Remove apostrophes
        if "'" in current_name:
            violations.append("Apostrophes should be removed")
            current_name = self._remove_apostrophes(current_name)

        # Rule 4: Check for special characters (except parentheses in govt entities)
        is_govt_entity = self._is_government_entity(current_name)
        if not is_govt_entity and self._has_special_characters(current_name):
            violations.append("Special characters should be removed")
            current_name = self._remove_special_characters(current_name, is_govt_entity)

        # Rule 5: Remove business designations
        has_designation = self._has_business_designation(current_name)
        if has_designation:
            violations.append("Business designations should be removed")
            current_name = self._remove_business_designations(current_name)

        # Rule 6: Expand company abbreviations
        has_abbrev = self._has_abbreviation(current_name)
        if has_abbrev:
            violations.append("Abbreviations should be expanded")
            current_name = self._expand_abbreviations(current_name)

        # Rule 7: Official spellings (would require a database of correct spellings)
        # Not implemented as it requires specific knowledge

        # Rule 8: Government entity formatting
        govt_entity_issue = self._check_government_entity_format(current_name)
        if govt_entity_issue:
            violations.append(govt_entity_issue)
            current_name = self._format_government_entity(current_name)

        # Rule 9: Apply proper title case with exceptions for small words
        if not self._check_title_case(current_name):
            violations.append(
                "Entity name should use proper title case (small words like 'and', 'of', 'to' should not be capitalized in the middle)"
            )
            current_name = self._proper_title_case(current_name)

        # If no changes were made, clear the violations list
        # If only violation is that it's an acronym, clear the suggestion
        if violations == [
            "Entity appears to be an acronym - full name should be spelled out"
        ]:
            return "", violations

        # If no changes were made, clear the violations list (except for acronym detection)
        if original_name == current_name and not self._is_likely_acronym(original_name):
            violations = []

        return current_name, violations

    def _replace_ampersands(self, name):
        """Replace '&' with 'and'."""
        return name.replace("&", "and")

    def _remove_apostrophes(self, name):
        """Remove apostrophes from the name."""
        return name.replace("'", "")

    def _has_special_characters(self, name):
        """Check if name contains special characters."""
        # Define characters to allow (alphanumeric, spaces, and parentheses for state codes)
        allowed_chars = set(string.ascii_letters + string.digits + " ()")
        return any(char not in allowed_chars for char in name)

    def _remove_special_characters(self, name, is_govt_entity=False):
        """
        Remove special characters from name.

        If is_govt_entity is True, preserve parentheses for state codes.
        """
        if is_govt_entity:
            # For government entities, keep parentheses for state codes
            # but remove other special characters
            pattern = r"[^\w\s\(\)]"
        else:
            # For non-government entities, remove all special characters
            pattern = r"[^\w\s]"

        return re.sub(pattern, "", name)

    def _has_business_designation(self, name):
        """Check if name contains business designations."""
        for designation in self.BUSINESS_DESIGNATIONS:
            # Match whole words only with word boundaries
            pattern = r"\b" + re.escape(designation) + r"\b"
            if re.search(pattern, name):
                return True
        return False

    def _remove_business_designations(self, name):
        """Remove business designations from name."""
        result = name
        for designation in self.BUSINESS_DESIGNATIONS:
            # Match whole words only with word boundaries
            pattern = r"\b" + re.escape(designation) + r"\b"
            result = re.sub(pattern, "", result).strip()
            # Remove any double spaces that might result
            result = re.sub(r"\s+", " ", result).strip()
        return result

    def _has_abbreviation(self, name):
        """Check if name contains known abbreviations."""
        for abbrev in self.ABBREVIATIONS:
            # Match whole words with case-insensitivity
            pattern = r"\b" + re.escape(abbrev) + r"\b"
            if re.search(pattern, name, re.IGNORECASE):
                return True
        return False

    def _expand_abbreviations(self, name):
        """Expand abbreviations in name."""
        result = name
        for abbrev, expansion in self.ABBREVIATIONS.items():
            # Match whole words with case-insensitivity
            pattern = r"\b" + re.escape(abbrev) + r"\b"

            # Find all matches and preserve case
            matches = re.finditer(pattern, result, re.IGNORECASE)

            # Process matches from end to start to avoid index shifting
            matches = list(matches)
            for match in reversed(matches):
                start, end = match.span()
                matched_text = result[start:end]

                # Determine the case of the replacement
                if matched_text.isupper():
                    replacement = expansion.upper()
                elif matched_text.istitle():
                    replacement = expansion.title()
                else:
                    replacement = expansion

                # Replace the matched text with the appropriate case
                result = result[:start] + replacement + result[end:]

        return result

    def _is_government_entity(self, name):
        """Check if name is a government entity."""
        for pattern in self.GOVT_ENTITY_PATTERNS.values():
            if re.match(pattern, name, re.IGNORECASE):
                return True
        return False

    def _check_government_entity_format(self, name):
        """Check if government entity name follows the correct format."""
        # Use case-insensitive matching for all patterns
        county_match = re.match(
            r"(?i)^County of ([A-Za-z ]+)(\s*[-]?\s*\([A-Z]{2}\))?$", name
        )
        if county_match and not county_match.group(2):
            return "County must include state in parentheses"

        city_match = re.match(
            r"(?i)^(City|Town|Village|Borough) of ([A-Za-z ]+)(\s*[-]?\s*\([A-Z]{2}\))?$",
            name,
        )
        if city_match and not city_match.group(3):
            return f"{city_match.group(1).title()} must include state in parentheses"

        state_code_match = re.search(r"\(([A-Z]{2})\)", name)
        if state_code_match and state_code_match.group(1) not in self.US_STATES:
            return f"Invalid state code: {state_code_match.group(1)}"

        return None

    def _format_government_entity(self, name):
        """Format government entity according to rules."""
        # State formatting
        state_match = re.match(r"(?i)^State of ([A-Za-z ]+)$", name)
        if state_match:
            state_name = state_match.group(1).strip()
            return f"State of {state_name.title()}"

        # County formatting
        county_match = re.match(
            r"(?i)^County of ([A-Za-z ]+)(\s*[-]?\s*\([A-Z]{2}\))?$", name
        )
        if county_match:
            county_name = county_match.group(1).strip()

            # Extract state code if present
            state_code = ""
            if county_match.group(2):
                state_code_match = re.search(r"\(([A-Z]{2})\)", county_match.group(2))
                if state_code_match:
                    state_code = state_code_match.group(1)

            # Format with state code
            if state_code:
                return f"County of {county_name.title()} ({state_code})"
            else:
                return f"County of {county_name.title()}"

        # City/town formatting
        city_match = re.match(
            r"(?i)^(City|Town|Village|Borough) of ([A-Za-z ]+)(\s*[-]?\s*\([A-Z]{2}\))?$",
            name,
        )
        if city_match:
            entity_type = city_match.group(1).strip().title()
            city_name = city_match.group(2).strip()

            # Extract state code if present
            state_code = ""
            if city_match.group(3):
                state_code_match = re.search(r"\(([A-Z]{2})\)", city_match.group(3))
                if state_code_match:
                    state_code = state_code_match.group(1)

            # Format with state code
            if state_code:
                return f"{entity_type} of {city_name.title()} ({state_code})"
            else:
                return f"{entity_type} of {city_name.title()}"

        return name

    def _is_likely_acronym(self, name):
        """Check if a name is likely an acronym.

        Args:
            name (str): The entity name to check

        Returns:
            bool: True if the name appears to be an acronym
        """
        # Skip empty names
        if not name or not isinstance(name, str):
            return False

        # Skip known government entity patterns
        if self._is_government_entity(name):
            return False

        # Check if it's entirely uppercase with few words
        words = name.split()
        if len(words) <= 3 and name.isupper():
            # Check for common non-acronym patterns
            if any(len(word) >= 8 for word in words):  # Long words likely not acronyms
                return False

            # Check if it contains any known abbreviations
            # This helps avoid flagging things we can already fix
            if self._has_abbreviation(name):
                return False

            # Otherwise, it's likely an acronym
            return True

        # Check for single words that are all uppercase with 2-6 letters (common acronym pattern)
        if len(words) == 1 and name.isupper() and 2 <= len(name) <= 6:
            return True

        return False

    def set_fuzzy_threshold(self, threshold):
        """
        Set the fuzzy matching threshold.

        Args:
            threshold (int): Value between 0 and 100.

        Raises:
            ValueError: If threshold is not between 0 and 100.
        """
        if not 0 <= threshold <= 100:
            raise ValueError("Fuzzy matching threshold must be between 0 and 100")
        self.fuzzy_threshold = threshold

    def get_fuzzy_threshold(self):
        """Get the current fuzzy matching threshold."""
        return self.fuzzy_threshold

    def _proper_title_case(self, name):
        """
        Apply proper title case with exceptions for small words in the middle.
        First and last words are always capitalized regardless of length.
        """
        # List of words that should not be capitalized when in the middle
        small_words = {
            "a",
            "an",
            "the",
            "and",
            "but",
            "or",
            "nor",
            "for",
            "so",
            "yet",
            "at",
            "by",
            "from",
            "in",
            "of",
            "on",
            "to",
            "with",
            "as",
        }

        # First convert entire string to title case
        result = name.title()

        # Split into words
        words = result.split()

        # Keep first and last words capitalized, lowercase small words in the middle
        for i in range(1, len(words) - 1):
            if words[i].lower() in small_words:
                words[i] = words[i].lower()

        # Rejoin with spaces
        return " ".join(words)

    def _check_title_case(self, name):
        """Check if name follows proper title case rules."""
        return name == self._proper_title_case(name)
