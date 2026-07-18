import re

import streamlit as st
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI MCQ Quality Analyzer",
    page_icon="✅",
    layout="centered"
)

st.title("AI MCQ Quality Analyzer")
st.caption("A lightweight NLP-based MCQ evaluation tool")

st.write(
    "Enter an MCQ to evaluate its structure, readability, "
    "difficulty and option quality."
)

question = st.text_area(
    "Enter your question",
    placeholder="Example: What is the capital of Pakistan?"
)

option_a = st.text_input("Option A")
option_b = st.text_input("Option B")
option_c = st.text_input("Option C")
option_d = st.text_input("Option D")

correct_answer = st.selectbox(
    "Select the correct answer",
    ["Option A", "Option B", "Option C", "Option D"]
)

subject = st.text_input(
    "Subject",
    placeholder="Example: General Knowledge"
)

grade_level = st.selectbox(
    "Grade level",
    [
        "Primary",
        "Middle",
        "Secondary",
        "Higher Education"
    ]
)


def identify_bloom_level(question_text):
    bloom_verbs = {
        "Remember": [
            "define",
            "identify",
            "list",
            "name",
            "recall",
            "state",
            "what",
            "when",
            "where",
            "who"
        ],
        "Understand": [
            "classify",
            "describe",
            "discuss",
            "explain",
            "interpret",
            "summarize"
        ],
        "Apply": [
            "apply",
            "calculate",
            "demonstrate",
            "determine",
            "solve",
            "use"
        ],
        "Analyze": [
            "analyze",
            "compare",
            "contrast",
            "differentiate",
            "examine",
            "investigate"
        ],
        "Evaluate": [
            "assess",
            "critique",
            "defend",
            "evaluate",
            "judge",
            "justify"
        ],
        "Create": [
            "construct",
            "create",
            "design",
            "develop",
            "formulate",
            "propose"
        ]
    }

    words = set(
        re.findall(
            r"\b[a-zA-Z]+\b",
            question_text.lower()
        )
    )

    for level, verbs in bloom_verbs.items():
        if words.intersection(verbs):
            return level

    return "Not clearly identified"


def calculate_option_similarity(options):
    try:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4)
        )

        vectors = vectorizer.fit_transform(options)
        similarity_matrix = cosine_similarity(vectors)

        similarities = []

        for row in range(len(options)):
            for column in range(row + 1, len(options)):
                similarities.append(
                    similarity_matrix[row][column]
                )

        if similarities:
            return max(similarities)

        return 0.0

    except ValueError:
        return 0.0


def estimate_difficulty(
    question_text,
    options,
    bloom_level
):
    question_word_count = len(
        question_text.split()
    )

    average_option_words = sum(
        len(option.split())
        for option in options
    ) / len(options)

    higher_order_levels = [
        "Analyze",
        "Evaluate",
        "Create"
    ]

    if bloom_level in higher_order_levels:
        return "Hard"

    if (
        question_word_count > 25
        or average_option_words > 8
    ):
        return "Hard"

    if (
        question_word_count > 12
        or average_option_words > 4
    ):
        return "Moderate"

    return "Easy"


def analyze_mcq(
    question_text,
    options,
    selected_grade
):
    score = 100

    strengths = []
    warnings = []
    suggestions = []

    clean_question = question_text.strip()

    clean_options = [
        option.strip()
        for option in options
    ]

    question_words = clean_question.split()
    word_count = len(question_words)

    # Question-length analysis
    if word_count < 5:
        score -= 15

        warnings.append(
            "The question may be too short or unclear."
        )

        suggestions.append(
            "Add enough context to make the question clear."
        )

    elif word_count > 40:
        score -= 10

        warnings.append(
            "The question is unnecessarily long."
        )

        suggestions.append(
            "Remove words that are not needed."
        )

    else:
        strengths.append(
            "The question length is appropriate."
        )

    # Question-mark analysis
    if not clean_question.endswith("?"):
        score -= 5

        warnings.append(
            "The question does not end with a question mark."
        )

        suggestions.append(
            "Add a question mark at the end."
        )

    # Exact duplicate-option analysis
    normalized_options = [
        re.sub(
            r"\s+",
            " ",
            option.lower()
        )
        for option in clean_options
    ]

    has_duplicate_options = (
        len(set(normalized_options)) < 4
    )

    if has_duplicate_options:
        score -= 25

        warnings.append(
            "Two or more options are identical."
        )

        suggestions.append(
            "Make every option unique."
        )

    else:
        strengths.append(
            "All four options are unique."
        )

    # Option-length analysis
    option_lengths = [
        len(option.split())
        for option in clean_options
    ]

    if (
        max(option_lengths)
        - min(option_lengths)
        > 5
    ):
        score -= 10

        warnings.append(
            "The option lengths are noticeably unbalanced."
        )

        suggestions.append(
            "Keep options similar in length so the "
            "correct answer is not obvious."
        )

    else:
        strengths.append(
            "The option lengths are reasonably balanced."
        )

    # Negative-wording analysis
    negative_pattern = (
        r"\b(not|except|never|incorrect|least)\b"
    )

    if re.search(
        negative_pattern,
        clean_question.lower()
    ):
        score -= 10

        warnings.append(
            "The question uses potentially confusing "
            "negative wording."
        )

        suggestions.append(
            "Rewrite it positively or clearly emphasize "
            "the negative word."
        )

    else:
        strengths.append(
            "The question avoids confusing negative wording."
        )

    # Weak-option analysis
    weak_options = [
        "all of the above",
        "none of the above",
        "both a and b",
        "all are correct"
    ]

    has_weak_option = any(
        option.lower() in weak_options
        for option in clean_options
    )

    if has_weak_option:
        score -= 10

        warnings.append(
            "A weak general-purpose option was detected."
        )

        suggestions.append(
            "Replace it with a meaningful distractor."
        )

    # Near-duplicate semantic analysis
    maximum_similarity = (
        calculate_option_similarity(
            clean_options
        )
    )

    if (
        maximum_similarity >= 0.85
        and not has_duplicate_options
    ):
        score -= 10

        warnings.append(
            "Two options appear extremely similar."
        )

        suggestions.append(
            "Rewrite similar options so each one "
            "represents a distinct choice."
        )

    elif not has_duplicate_options:
        strengths.append(
            "No near-duplicate options were detected."
        )

    # Readability analysis
    readability_grade = max(
        0.0,
        round(
            textstat.flesch_kincaid_grade(
                clean_question
            ),
            1
        )
    )

    grade_limits = {
        "Primary": 6,
        "Middle": 9,
        "Secondary": 12,
        "Higher Education": 20
    }

    # Readability formulas are unreliable
    # for extremely short questions.
    if (
        word_count >= 10
        and readability_grade
        > grade_limits[selected_grade]
    ):
        score -= 10

        warnings.append(
            f"The wording may be difficult for "
            f"the selected {selected_grade} level."
        )

        suggestions.append(
            "Use shorter sentences and simpler "
            "vocabulary for the target learners."
        )

    else:
        strengths.append(
            "The readability suits the selected "
            "grade category."
        )

    bloom_level = identify_bloom_level(
        clean_question
    )

    difficulty = estimate_difficulty(
        clean_question,
        clean_options,
        bloom_level
    )

    score = max(
        0,
        min(score, 100)
    )

    return {
        "score": score,
        "strengths": strengths,
        "warnings": warnings,
        "suggestions": suggestions,
        "bloom_level": bloom_level,
        "difficulty": difficulty,
        "readability_grade": readability_grade,
        "maximum_similarity": maximum_similarity
    }


if st.button(
    "Analyze MCQ",
    type="primary"
):
    options = [
        option_a,
        option_b,
        option_c,
        option_d
    ]

    if (
        not question.strip()
        or not all(
            option.strip()
            for option in options
        )
    ):
        st.error(
            "Please enter the question and "
            "all four options."
        )

    else:
        result = analyze_mcq(
            question,
            options,
            grade_level
        )

        st.divider()
        st.subheader("Analysis Results")

        column_1, column_2, column_3 = st.columns(3)

        column_1.metric(
            "Quality Score",
            f"{result['score']}/100"
        )

        column_2.metric(
            "Difficulty",
            result["difficulty"]
        )

        column_3.metric(
            "Readability Grade",
            result["readability_grade"]
        )

        st.write(
            f"**Bloom’s Taxonomy:** "
            f"{result['bloom_level']}"
        )

        st.write(
            f"**Selected Answer:** "
            f"{correct_answer}"
        )

        st.write(
            f"**Subject:** "
            f"{subject or 'Not provided'}"
        )

        st.write(
            f"**Target Level:** "
            f"{grade_level}"
        )

        if result["score"] >= 80:
            st.success(
                "Good-quality MCQ"
            )

        elif result["score"] >= 60:
            st.warning(
                "This MCQ needs some improvement"
            )

        else:
            st.error(
                "This MCQ needs major improvement"
            )

        if result["strengths"]:
            st.subheader("Strengths")

            for strength in result["strengths"]:
                st.write(
                    f"✅ {strength}"
                )

        if result["warnings"]:
            st.subheader("Issues Detected")

            for warning in result["warnings"]:
                st.write(
                    f"⚠️ {warning}"
                )

        if result["suggestions"]:
            st.subheader(
                "Improvement Suggestions"
            )

            for suggestion in result["suggestions"]:
                st.write(
                    f"💡 {suggestion}"
                )

        with st.expander(
            "Technical NLP Details"
        ):
            st.write(
                "Maximum similarity between "
                "any two options:",
                f"{result['maximum_similarity']:.2f}"
            )

            st.write(
                "Similarity method:",
                "Character-level TF-IDF "
                "with cosine similarity"
            )

            st.write(
                "Readability method:",
                "Flesch–Kincaid grade level"
            )

            if len(question.split()) < 10:
                st.info(
                    "The readability score is displayed, "
                    "but no readability penalty was applied "
                    "because the question is very short."
                )