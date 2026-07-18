# AI MCQ Quality Analyzer

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://testinflow-mcq-analyzer.streamlit.app)

## Live Demo

Try the application online:

**[Launch AI MCQ Quality Analyzer](https://testinflow-mcq-analyzer.streamlit.app)**

A lightweight NLP-based web application for evaluating the structural quality, readability, difficulty, Bloom’s taxonomy level, and distractor similarity of multiple-choice questions.

The project is being developed as part of the **TestInFlow AI Lab** to support teachers in designing clearer and more effective assessments.

## Project Purpose

Writing a good MCQ involves more than providing one correct answer and three incorrect options. Questions may contain ambiguity, duplicate choices, weak distractors, unsuitable vocabulary, negative wording, or unbalanced option lengths.

This application automatically examines an MCQ and provides an explainable quality report.

## Current Features

- Overall MCQ quality score
- Question-length analysis
- Missing question-mark detection
- Exact duplicate-option detection
- Near-duplicate option detection
- Option-length balance analysis
- Negative-wording detection
- Weak general-purpose option detection
- Flesch–Kincaid readability grade
- Estimated question difficulty
- Bloom’s taxonomy estimation
- Strengths, warnings, and improvement suggestions
- Technical NLP details for transparency

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- TF-IDF vectorization
- Cosine similarity
- Textstat
- Regular expressions
- Git and GitHub

## NLP Methods

### Option Similarity

Character-level TF-IDF converts the answer options into numerical vectors. Cosine similarity is then used to identify options that may be extremely similar.

### Readability Analysis

The Flesch–Kincaid grade-level formula estimates the educational reading level of the question.

### Bloom’s Taxonomy

The current version uses educational action verbs to estimate whether a question targets remembering, understanding, applying, analyzing, evaluating, or creating.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Mtahirdlbar/mcq-quality-analyzer.git