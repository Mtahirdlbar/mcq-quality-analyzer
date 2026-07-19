import re

import pandas as pd
import streamlit as st
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from clarity_analyzer import analyze_grammar_and_clarity


st.set_page_config(
    page_title="AI MCQ Quality Analyzer",
    page_icon=":material/quiz:",
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

    clarity_result = analyze_grammar_and_clarity(
        clean_question
    )

    score -= clarity_result["penalty"]

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
        "maximum_similarity": maximum_similarity,
        "clarity_result": clarity_result
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

        column_1, column_2, column_3, column_4 = st.columns(4)

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

        column_4.metric(
            "Grammar & Clarity",
            f"{100 - result['clarity_result']['penalty']}/100"
        )

        st.write(
            f"**Bloom's Taxonomy:** "
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
                    f"[OK] {strength}"
                )

        if result["warnings"]:
            st.subheader("Issues Detected")

            for warning in result["warnings"]:
                st.write(
                    f"[ISSUE] {warning}"
                )

        if result["suggestions"]:
            st.subheader(
                "Improvement Suggestions"
            )

            for suggestion in result["suggestions"]:
                st.write(
                    f"[TIP] {suggestion}"
                )

        st.subheader("Grammar & Clarity Analysis")

        clarity_result = result["clarity_result"]

        if clarity_result["issues"]:
            for issue in clarity_result["issues"]:
                st.write(
                    f"[ISSUE] {issue}"
                )

            st.write("**Grammar and clarity suggestions:**")

            for suggestion in clarity_result["suggestions"]:
                st.write(
                    f"[TIP] {suggestion}"
                )

        else:
            st.success(
                "No basic grammar or clarity problems were detected."
            )

        if clarity_result["strengths"]:
            with st.expander("Grammar and clarity strengths"):
                for strength in clarity_result["strengths"]:
                    st.write(
                        f"[OK] {strength}"
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
                "Flesch-Kincaid grade level"
            )

            st.write(
                "Grammar and clarity method:",
                "Explainable heuristic language checks"
            )

            if len(question.split()) < 10:
                st.info(
                    "The readability score is displayed, "
                    "but no readability penalty was applied "
                    "because the question is very short."
                )


st.divider()
st.header("Batch MCQ Analysis")
st.write(
    "Upload a CSV file to evaluate multiple MCQs and download "
    "a complete quality report."
)

required_columns = [
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "correct_answer",
    "subject",
    "grade_level"
]

st.caption(
    "Required columns: "
    + ", ".join(required_columns)
)

uploaded_file = st.file_uploader(
    "Upload MCQ CSV file",
    type=["csv"]
)


def clean_csv_value(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(
            "The CSV file could not be read. Please use a valid CSV file."
        )
        st.caption(f"Technical detail: {error}")
    else:
        batch_data.columns = [
            str(column).strip().lower()
            for column in batch_data.columns
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in batch_data.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
        elif batch_data.empty:
            st.warning("The uploaded CSV file does not contain any MCQs.")
        else:
            st.success(
                f"CSV loaded successfully: {len(batch_data)} MCQ(s) found."
            )

            with st.expander("Preview uploaded MCQs"):
                st.dataframe(
                    batch_data,
                    use_container_width=True
                )

            if st.button(
                "Analyze Uploaded MCQs",
                type="primary"
            ):
                allowed_grades = {
                    "primary": "Primary",
                    "middle": "Middle",
                    "secondary": "Secondary",
                    "higher education": "Higher Education"
                }

                allowed_answers = {
                    "a": "Option A",
                    "b": "Option B",
                    "c": "Option C",
                    "d": "Option D",
                    "option a": "Option A",
                    "option b": "Option B",
                    "option c": "Option C",
                    "option d": "Option D"
                }

                analysis_rows = []

                for index, row in batch_data.iterrows():
                    batch_question = clean_csv_value(
                        row["question"]
                    )

                    batch_options = [
                        clean_csv_value(row["option_a"]),
                        clean_csv_value(row["option_b"]),
                        clean_csv_value(row["option_c"]),
                        clean_csv_value(row["option_d"])
                    ]

                    batch_answer = clean_csv_value(
                        row["correct_answer"]
                    )
                    normalized_answer = batch_answer.lower()
                    batch_subject = clean_csv_value(
                        row["subject"]
                    )
                    raw_grade = clean_csv_value(
                        row["grade_level"]
                    )
                    normalized_grade = raw_grade.lower()

                    row_number = index + 2

                    if not batch_question or not all(batch_options):
                        analysis_rows.append({
                            "row_number": row_number,
                            "question": batch_question,
                            "subject": batch_subject,
                            "grade_level": raw_grade,
                            "correct_answer": batch_answer,
                            "status": "Invalid: missing question or option",
                            "quality_score": None,
                            "grammar_clarity_score": None,
                            "difficulty": "",
                            "readability_grade": None,
                            "bloom_level": "",
                            "maximum_option_similarity": None,
                            "issues": "",
                            "suggestions": ""
                        })
                        continue

                    if normalized_answer not in allowed_answers:
                        analysis_rows.append({
                            "row_number": row_number,
                            "question": batch_question,
                            "subject": batch_subject,
                            "grade_level": raw_grade,
                            "correct_answer": batch_answer,
                            "status": "Invalid correct answer",
                            "quality_score": None,
                            "grammar_clarity_score": None,
                            "difficulty": "",
                            "readability_grade": None,
                            "bloom_level": "",
                            "maximum_option_similarity": None,
                            "issues": "",
                            "suggestions": ""
                        })
                        continue

                    if normalized_grade not in allowed_grades:
                        analysis_rows.append({
                            "row_number": row_number,
                            "question": batch_question,
                            "subject": batch_subject,
                            "grade_level": raw_grade,
                            "correct_answer": batch_answer,
                            "status": "Invalid grade level",
                            "quality_score": None,
                            "grammar_clarity_score": None,
                            "difficulty": "",
                            "readability_grade": None,
                            "bloom_level": "",
                            "maximum_option_similarity": None,
                            "issues": "",
                            "suggestions": ""
                        })
                        continue

                    selected_grade = allowed_grades[normalized_grade]
                    batch_answer = allowed_answers[normalized_answer]
                    batch_result = analyze_mcq(
                        batch_question,
                        batch_options,
                        selected_grade
                    )

                    combined_issues = (
                        batch_result["warnings"]
                        + batch_result["clarity_result"]["issues"]
                    )
                    combined_suggestions = (
                        batch_result["suggestions"]
                        + batch_result["clarity_result"]["suggestions"]
                    )

                    analysis_rows.append({
                        "row_number": row_number,
                        "question": batch_question,
                        "subject": batch_subject,
                        "grade_level": selected_grade,
                        "correct_answer": batch_answer,
                        "status": "Analyzed",
                        "quality_score": batch_result["score"],
                        "grammar_clarity_score": (
                            100
                            - batch_result["clarity_result"]["penalty"]
                        ),
                        "difficulty": batch_result["difficulty"],
                        "readability_grade": batch_result[
                            "readability_grade"
                        ],
                        "bloom_level": batch_result["bloom_level"],
                        "maximum_option_similarity": round(
                            batch_result["maximum_similarity"],
                            3
                        ),
                        "issues": " | ".join(combined_issues),
                        "suggestions": " | ".join(
                            combined_suggestions
                        )
                    })

                report_data = pd.DataFrame(analysis_rows)
                valid_results = report_data[
                    report_data["status"] == "Analyzed"
                ]

                st.subheader("Batch Analysis Summary")

                if valid_results.empty:
                    st.error(
                        "No valid MCQs were available for analysis."
                    )
                else:
                    summary_1, summary_2, summary_3, summary_4 = (
                        st.columns(4)
                    )

                    summary_1.metric(
                        "MCQs Analyzed",
                        len(valid_results)
                    )
                    summary_2.metric(
                        "Average Quality",
                        f"{valid_results['quality_score'].mean():.1f}/100"
                    )
                    summary_3.metric(
                        "Good-quality MCQs",
                        int(
                            (
                                valid_results["quality_score"] >= 80
                            ).sum()
                        )
                    )
                    summary_4.metric(
                        "Need Improvement",
                        int(
                            (
                                valid_results["quality_score"] < 80
                            ).sum()
                        )
                    )

                st.dataframe(
                    report_data,
                    use_container_width=True
                )

                report_csv = report_data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Analysis Report",
                    data=report_csv,
                    file_name="mcq_quality_analysis_report.csv",
                    mime="text/csv"
                )