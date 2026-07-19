# AI MCQ Quality Analyzer

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://testinflow-mcq-analyzer.streamlit.app)

A lightweight, explainable NLP application for evaluating the quality of multiple-choice questions. It supports both individual MCQ analysis and batch analysis through CSV upload.

Developed as part of the **TestInFlow AI Lab** to help teachers create clearer and more effective assessments.

## Live Demo

**[Launch AI MCQ Quality Analyzer](https://testinflow-mcq-analyzer.streamlit.app)**

## Project Purpose

Writing a good MCQ involves more than providing one correct answer and three incorrect options. Questions may contain ambiguity, duplicate choices, weak distractors, unsuitable vocabulary, negative wording, unbalanced option lengths, or grammar and clarity problems.

This application examines these characteristics and generates an explainable quality report.

## Current Features

- Overall MCQ quality score
- Grammar and clarity score
- Question-length analysis
- Missing question-mark detection
- Exact duplicate-option detection
- Near-duplicate option detection
- Option-length balance analysis
- Negative-wording detection
- Weak general-purpose option detection
- Flesch-Kincaid readability grade
- Estimated question difficulty
- Bloom's taxonomy estimation
- Explainable strengths, issues, and suggestions
- Technical NLP details for transparency
- Batch MCQ analysis through CSV upload
- Batch summary statistics
- Downloadable CSV analysis report
- Invalid CSV and missing-column validation

## Batch CSV Format

The batch analyzer requires the following columns:

```text
question,option_a,option_b,option_c,option_d,correct_answer,subject,grade_level
```

Supported grade levels are:

- Primary
- Middle
- Secondary
- Higher Education

A ready-to-use example is available in `sample_mcqs.csv`.

## Technologies Used

- Python
- Streamlit
- Pandas
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

The Flesch-Kincaid grade-level formula estimates the educational reading level of the question.

### Bloom's Taxonomy

The current version uses educational action verbs to estimate whether a question targets remembering, understanding, applying, analyzing, evaluating, or creating.

### Grammar and Clarity

Explainable heuristic checks identify issues such as lowercase openings, repeated words, punctuation spacing, double negatives, vague wording, complex sentence structure, and absolute wording.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Mtahirdlbar/mcq-quality-analyzer.git
cd mcq-quality-analyzer
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
streamlit run app.py
```

## Current Limitations

- The system is an explainable quality-support tool, not a replacement for expert educational judgment.
- Bloom's taxonomy and difficulty are currently estimated using transparent heuristic rules.
- Option similarity currently focuses mainly on lexical patterns; deeper semantic evaluation is planned.

## Author

**Muhammad Tahir Dilbar**  
AI Lecturer and PhD Researcher  
TestInFlow AI Lab

- Website: [TestInFlow](https://testinflow.com)
- GitHub: [Mtahirdlbar](https://github.com/Mtahirdlbar)
