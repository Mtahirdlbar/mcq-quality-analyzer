import re


def analyze_grammar_and_clarity(question_text):
    """
    Perform lightweight, explainable grammar and clarity checks.

    This is a heuristic baseline, not a complete grammar-correction model.
    """

    question = question_text.strip()

    issues = []
    suggestions = []
    strengths = []
    penalty = 0

    words = re.findall(
        r"\b[\w'-]+\b",
        question.lower()
    )

    # Capital-letter check
    if question and question[0].islower():
        penalty += 5

        issues.append(
            "The question begins with a lowercase letter."
        )

        suggestions.append(
            "Begin the question with a capital letter."
        )

    else:
        strengths.append(
            "The question begins with appropriate capitalization."
        )

    # Repeated consecutive words
    repeated_words = re.findall(
        r"\b(\w+)\s+\1\b",
        question,
        flags=re.IGNORECASE
    )

    if repeated_words:
        penalty += 10

        repeated_text = ", ".join(
            sorted(set(repeated_words))
        )

        issues.append(
            f"Repeated consecutive word detected: {repeated_text}."
        )

        suggestions.append(
            "Remove the accidentally repeated word."
        )

    else:
        strengths.append(
            "No repeated consecutive words were detected."
        )

    # Extra-space check
    if re.search(r"\s{2,}", question):
        penalty += 3

        issues.append(
            "The question contains unnecessary extra spaces."
        )

        suggestions.append(
            "Replace multiple spaces with a single space."
        )

    # Space before punctuation
    if re.search(r"\s+[?.!,;:]", question):
        penalty += 3

        issues.append(
            "A space appears before punctuation."
        )

        suggestions.append(
            "Remove spaces that appear before punctuation marks."
        )

    # Missing space after punctuation
    if re.search(r"[,:;.!?][A-Za-z]", question):
        penalty += 3

        issues.append(
            "A punctuation mark is not followed by a space."
        )

        suggestions.append(
            "Add a space after the punctuation mark."
        )

    # Double-negative check
    negative_words = re.findall(
        r"\b(not|never|no|none|neither|without)\b",
        question.lower()
    )

    if len(negative_words) >= 2:
        penalty += 10

        issues.append(
            "The question may contain a confusing double negative."
        )

        suggestions.append(
            "Rewrite the question using one clear negative expression."
        )

    # Vague reference check
    vague_openings = (
        "this is",
        "that is",
        "it is",
        "they are",
        "these are",
        "those are"
    )

    if question.lower().startswith(vague_openings):
        penalty += 8

        issues.append(
            "The question begins with a potentially vague reference."
        )

        suggestions.append(
            "Name the exact concept instead of using this, that, it or they."
        )

    # Overly complex sentence check
    comma_count = question.count(",")
    conjunction_count = sum(
        words.count(word)
        for word in ["and", "but", "although", "because", "while"]
    )

    if len(words) > 30 and (
        comma_count >= 3
        or conjunction_count >= 3
    ):
        penalty += 10

        issues.append(
            "The question may contain too many ideas in one sentence."
        )

        suggestions.append(
            "Split or simplify the sentence so it tests one clear idea."
        )

    # Absolute wording check
    absolute_words = [
        word for word in [
            "always",
            "never",
            "completely",
            "only",
            "all",
            "none"
        ]
        if word in words
    ]

    if absolute_words:
        penalty += 5

        issues.append(
            "Absolute wording may make the answer easier to guess."
        )

        suggestions.append(
            "Use absolute terms only when they are academically necessary."
        )

    # Clarity strength
    if not issues:
        strengths.append(
            "No basic grammar or clarity problems were detected."
        )

    return {
        "penalty": min(penalty, 25),
        "issues": issues,
        "suggestions": suggestions,
        "strengths": strengths
    }