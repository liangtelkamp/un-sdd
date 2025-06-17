import logging
from typing import List, Tuple, Union

try:
    from presidio_analyzer import AnalyzerEngine
except ImportError:
    raise ImportError("presidio-analyzer is not installed. Install with: pip install presidio-analyzer")

import pandas as pd

logger = logging.getLogger(__name__)

class PresidioPIIDetector:
    """
    Detect PII in free text using Microsoft Presidio.
    """
    def __init__(self, language: str = "en"):
        self.analyzer = AnalyzerEngine()
        self.language = language

    def analyze_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Analyze a single string for PII.
        Returns (contains_pii: bool, pii_types: List[str])
        """
        if not isinstance(text, str) or not text.strip():
            return False, []
        results = self.analyzer.analyze(text=text, language=self.language)
        pii_types = list(set([r.entity_type for r in results]))
        return bool(pii_types), pii_types

    def analyze_series(self, series: pd.Series) -> pd.DataFrame:
        """
        Analyze a pandas Series (column) of free text.
        Returns a DataFrame with columns: [original_text, contains_pii, pii_types]
        """
        results = series.apply(self.analyze_text)
        contains_pii = results.apply(lambda x: x[0])
        pii_types = results.apply(lambda x: x[1])
        return pd.DataFrame({
            'original_text': series,
            'contains_pii': contains_pii,
            'pii_types': pii_types
        })

# Example usage
if __name__ == "__main__":
    detector = PresidioPIIDetector()
    sample_texts = [
        "My name is John Doe and my phone is 555-123-4567.",
        "No PII here!",
        "Contact: alice@example.com",
        "",  # empty
        None  # None
    ]
    import pandas as pd
    s = pd.Series(sample_texts)
    df = detector.analyze_series(s)
    print(df) 