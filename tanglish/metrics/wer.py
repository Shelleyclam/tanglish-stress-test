"""Word error rate (WER): how many words speech-to-text got wrong.

WER = (substituted + deleted + inserted words) / words in your transcript.
Both texts are normalised first (lowercase, no punctuation, small numbers as
words) so "2" vs "two" or a missing comma doesn't count as an error.
"""

import unicodedata

import jiwer

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    tens, ones = divmod(n, 10)
    return TENS[tens] + ("" if ones == 0 else " " + ONES[ones])


def _is_word_char(c: str) -> bool:
    """Letters, digits, ' and _, plus combining marks: Tamil vowel signs are marks, so
    dropping them would break கிடைக்குமா into fragments."""
    return c.isalnum() or c in "_'" or c.isspace() or unicodedata.category(c).startswith("M")


def normalize(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = "".join(c if _is_word_char(c) else " " for c in text)
    words = [number_to_words(int(w)) if w.isdigit() and int(w) < 100 else w
             for w in text.split()]
    return " ".join(words)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("reference transcript is empty")
    return round(jiwer.wer(ref, hyp), 3)
