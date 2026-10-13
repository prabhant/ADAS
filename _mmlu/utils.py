import random
import re
import string
from collections import namedtuple

import numpy as np

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

# Uppercase letters only, so an article ("a") or a word ("Apple") is never read as a choice.
_BARE_CHOICE = re.compile(r"^[\s\"'*`(\[]*([A-D])[\s\"'*`)\].:]*$")
_LEADING_CHOICE = re.compile(r"^[\s\"'*`]*[(\[]?([A-D])[)\].:,]")
# "A, B or D" names several options, so it is not one choice.
_SEVERAL_CHOICES = re.compile(r"^[\s\"'*`(\[]*[A-D][)\].:]?\s*(?:,|/|\bor\b|\band\b)\s*[(\[]?[A-D](?![A-Za-z])")
_STATED_CHOICE = re.compile(r"(?i:\b(?:answer|option|choice)\b)\s*(?:is|:|=)?\s*"
                            r"(?i:(?:option|choice)\s+)?[(\[]?([A-D])(?![A-Za-z])")


def _normalise(text):
    return re.sub(r"\W+", " ", str(text)).strip().lower()


def extract_choice(text, options=None):
    """Index 0-3 of the chosen option, or None when no single choice can be read.

    Stock ADAS accepts only a bare letter or "X)", which scores answers such as
    "D. Mass and acceleration", "(D)" or "The answer is D" as wrong.
    """
    text = str(text).strip()
    if _SEVERAL_CHOICES.search(text):
        return None
    for pattern in (_BARE_CHOICE, _LEADING_CHOICE, _STATED_CHOICE):
        match = pattern.search(text)
        if match:
            return "ABCD".index(match.group(1))
    if options:
        hits = [i for i, option in enumerate(options)
                if _normalise(option) and _normalise(option) == _normalise(text)]
        if len(hits) == 1:
            return hits[0]
    for index, letter in enumerate("ABCD"):  # the stock ADAS rule, last
        if f"{letter})" in text:
            return index
    return None

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question.

{Question}

(A) {A}
(B) {B}
(C) {C}
(D) {D}
""".strip()


def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"
