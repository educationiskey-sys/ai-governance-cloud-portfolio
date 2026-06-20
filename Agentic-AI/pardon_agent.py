"""
pardon_agent.py

This module provides a basic skeleton for an AI‑assisted workflow to help
attorneys process U.S. Department of Justice (DOJ) pardon petitions.
It illustrates how document ingestion, simple classification, and
extractive summarization could be combined to generate structured case
summaries for human review.  This code is **not** a complete production
system.  It intentionally avoids any automated decision‑making about
eligibility and must be used with a human in the loop.

Key features:
  * Document ingestion: Reads plain‑text files representing petition
    materials (court records, letters, etc.).  In a real system, this
    would handle PDFs, scanned documents, and structured data.
  * Simple classification: Uses keyword matching to categorize the
    offence type.  A production system would leverage trained models.
  * Extractive summarization: Implements a naive frequency‑based
    sentence selection algorithm to highlight key sentences.  This is
    purely demonstrative; consider using well‑tested libraries such
    as `sumy`, `transformers`, or `spacy` for more robust results.
  * Structured output: Produces a dictionary containing the original
    text, detected offence category, and a summary.  This can be
    serialized to JSON, sent to a user interface, or passed into
    downstream analytics.

Disclaimer:
    This code is for illustrative purposes only.  It should be
    reviewed, tested, and extended by qualified developers and legal
    experts before any operational use.  The authors assume no
    responsibility for any misuse or misinterpretation of this
    demonstration.
"""

import os
import re
import string
from collections import Counter
from typing import List, Dict


class DocumentIngestor:
    """Simple document reader for plain‑text files."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir

    def ingest(self, filename: str) -> str:
        """Read a text file from the configured directory.

        Args:
            filename: Name of the file to read.

        Returns:
            The file's contents as a string.

        Raises:
            FileNotFoundError: If the file cannot be found.
        """
        path = os.path.join(self.root_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {filename!r} does not exist in {self.root_dir!r}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


class CaseClassifier:
    """A rudimentary classifier that assigns an offence category based on keywords.

    This demonstration uses simple keyword searches.  A real system
    should employ machine learning models trained on annotated data.
    """

    # Mapping of offence categories to indicative keywords
    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "drug": ["drug", "controlled substance", "narcotic", "distribution"],
        "violent": ["assault", "murder", "homicide", "violent"],
        "financial": ["fraud", "embezzle", "white collar", "bank"],
        "weapon": ["firearm", "weapon", "gun", "possession"],
        "other": []
    }

    def classify(self, text: str) -> str:
        """Classify the offence type by searching for keywords.

        Args:
            text: The input document text.

        Returns:
            The name of the offence category.
        """
        lower_text = text.lower()
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in lower_text for kw in keywords):
                return category
        return "other"


class Summarizer:
    """Naive extractive summarizer using word frequency.

    This summarizer splits text into sentences, computes word
    frequencies, scores sentences by summing the frequencies of their
    words, and selects the top‑N sentences as the summary.  It
    performs basic preprocessing to remove punctuation and stop words.
    """

    # A minimal list of stop words; extend as needed.
    STOP_WORDS: set = {
        "the", "a", "an", "in", "on", "at", "by", "for", "and", "or", "to",
        "of", "with", "is", "are", "was", "were", "be", "been", "being",
    }

    def __init__(self, summary_sentences: int = 5) -> None:
        self.summary_sentences = summary_sentences

    @staticmethod
    def tokenize_sentences(text: str) -> List[str]:
        """Split text into sentences using regular expressions."""
        # Simple sentence splitter based on punctuation.  For more
        # accurate splitting, consider using NLTK's sent_tokenize or
        # spaCy's sentence segmentation.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s]

    @staticmethod
    def tokenize_words(text: str) -> List[str]:
        """Convert a sentence into lowercase words without punctuation."""
        return [word.lower().strip(string.punctuation) for word in text.split()]

    def summarize(self, text: str) -> str:
        """Produce an extractive summary of the given text.

        Args:
            text: The document to summarize.

        Returns:
            A string containing the top sentences concatenated.
        """
        sentences = self.tokenize_sentences(text)
        if not sentences:
            return ""

        # Compute word frequencies across the whole document
        words = []
        for sentence in sentences:
            words.extend(self.tokenize_words(sentence))
        frequencies = Counter(word for word in words if word and word not in self.STOP_WORDS)

        if not frequencies:
            # If no words left after removing stop words, return the first sentence
            return sentences[0]

        # Score sentences by summing word frequencies
        sentence_scores = []
        for sentence in sentences:
            sentence_words = self.tokenize_words(sentence)
            score = sum(frequencies.get(word, 0) for word in sentence_words)
            sentence_scores.append((score, sentence))

        # Select the top‑N sentences
        top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[: self.summary_sentences]
        # Preserve original order
        top_sentences_sorted = sorted(top_sentences, key=lambda x: sentences.index(x[1]))

        # Join the sentences to form the summary
        summary = " ".join(sentence for _, sentence in top_sentences_sorted)
        return summary


class PardonCaseAgent:
    """High‑level agent combining ingestion, classification, and summarization."""

    def __init__(self, data_dir: str, summary_sentences: int = 5) -> None:
        self.ingestor = DocumentIngestor(data_dir)
        self.classifier = CaseClassifier()
        self.summarizer = Summarizer(summary_sentences)

    def process_case(self, filename: str) -> Dict[str, str]:
        """Process a single petition file and return a structured result.

        Args:
            filename: Name of the petition file to process.

        Returns:
            A dictionary containing the original text, detected category
            and summary.
        """
        text = self.ingestor.ingest(filename)
        category = self.classifier.classify(text)
        summary = self.summarizer.summarize(text)
        return {
            "filename": filename,
            "category": category,
            "summary": summary,
            "full_text": text,
        }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Process DOJ pardon petition files.")
    parser.add_argument("data_dir", help="Directory containing petition text files")
    parser.add_argument("filename", help="Name of the file to process")
    parser.add_argument(
        "--sentences",
        type=int,
        default=5,
        help="Number of sentences to include in the summary (default: 5)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the structured JSON result. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    agent = PardonCaseAgent(args.data_dir, summary_sentences=args.sentences)
    result = agent.process_case(args.filename)

    # Serialize result to JSON
    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    main()