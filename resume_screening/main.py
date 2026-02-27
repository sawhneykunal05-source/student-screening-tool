import re
from pathlib import Path

import numpy as np
import pandas as pd


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "with", "will", "you", "your", "we", "our", "this",
    "role", "candidate", "required", "responsibilities", "experience", "years", "year", "skills",
}


def clean_text(text: str) -> str:
    """Convert text to lowercase, remove punctuation, and trim extra spaces."""
    if pd.isna(text):
        return ""

    cleaned = str(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_data(resume_file: str = "resumes.csv", job_file: str = "job_description.txt"):
    """Load resume CSV and job description text with basic validation and missing-value handling."""
    try:
        resumes_df = pd.read_csv(resume_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Resume file not found: {resume_file}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Resume file is empty: {resume_file}") from exc

    required_columns = [
        "candidate_name",
        "email",
        "skills",
        "experience_years",
        "education",
        "resume_text",
    ]

    missing_columns = [col for col in required_columns if col not in resumes_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in resumes.csv: {missing_columns}")

    # Handle missing values safely for text and numeric columns.
    text_columns = ["candidate_name", "email", "skills", "education", "resume_text"]
    for col in text_columns:
        resumes_df[col] = resumes_df[col].fillna("").astype(str)

    resumes_df["experience_years"] = pd.to_numeric(
        resumes_df["experience_years"], errors="coerce"
    ).fillna(0)

    # Clean free-text fields.
    resumes_df["resume_text"] = resumes_df["resume_text"].apply(clean_text)
    resumes_df["education"] = resumes_df["education"].apply(clean_text)

    # Convert comma-separated skills into normalized list format.
    resumes_df["skills"] = resumes_df["skills"].apply(
        lambda raw: [clean_text(skill) for skill in str(raw).split(",") if clean_text(skill)]
    )

    try:
        with open(job_file, "r", encoding="utf-8") as file:
            job_description = file.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Job description file not found: {job_file}") from exc

    return resumes_df, clean_text(job_description)


def extract_keywords(job_description: str):
    """Extract required skills keywords from job description text."""
    if not job_description:
        return set(), 5

    preferred_experience = 5
    years_match = re.search(r"(\d+)\s*\+?\s*years", job_description)
    if years_match:
        preferred_experience = int(years_match.group(1))

    # Prefer skills listed after "required skills" if the phrase exists.
    marker = "required skills"
    required_keywords = set()

    if marker in job_description:
        chunk = job_description.split(marker, maxsplit=1)[1]
        chunk = chunk.split("responsibilities", maxsplit=1)[0]
        for token in chunk.split():
            token = clean_text(token)
            if token and token not in STOPWORDS and len(token) > 1:
                required_keywords.add(token)

    # Fallback to all meaningful words in the job description.
    if not required_keywords:
        required_keywords = {
            word
            for word in job_description.split()
            if word not in STOPWORDS and len(word) > 1
        }

    return required_keywords, preferred_experience


def calculate_skill_score(candidate_skills, required_keywords):
    """Calculate skill match score from 0 to 100."""
    if not required_keywords:
        return 0.0

    candidate_set = set(candidate_skills)
    matched = len(candidate_set.intersection(required_keywords))
    return (matched / len(required_keywords)) * 100


def calculate_total_score(skill_scores, experience_years, preferred_experience):
    """Calculate weighted score using NumPy arrays."""
    skill_array = np.array(skill_scores, dtype=float)
    exp_years_array = np.array(experience_years, dtype=float)

    experience_scores = np.clip((exp_years_array / max(preferred_experience, 1)) * 100, 0, 100)
    total_scores = (0.6 * skill_array) + (0.4 * experience_scores)

    return experience_scores, total_scores


def rank_candidates(resumes_df, required_keywords, preferred_experience):
    """Compute scores, sort candidates, and assign rank."""
    resumes_df = resumes_df.copy()

    resumes_df["skill_match_score"] = resumes_df["skills"].apply(
        lambda candidate_skills: calculate_skill_score(candidate_skills, required_keywords)
    )

    experience_scores, total_scores = calculate_total_score(
        resumes_df["skill_match_score"].to_numpy(),
        resumes_df["experience_years"].to_numpy(),
        preferred_experience,
    )

    resumes_df["experience_score"] = np.round(experience_scores, 2)
    resumes_df["total_score"] = np.round(total_scores, 2)

    ranked_df = resumes_df.sort_values("total_score", ascending=False).reset_index(drop=True)
    ranked_df["rank"] = np.arange(1, len(ranked_df) + 1)

    return ranked_df


def save_results(ranked_df: pd.DataFrame, output_file: str = "screened_candidates.csv"):
    """Save selected ranking columns to CSV."""
    output_columns = [
        "candidate_name",
        "email",
        "skill_match_score",
        "experience_years",
        "total_score",
        "rank",
    ]

    final_df = ranked_df[output_columns].copy()
    final_df["skill_match_score"] = final_df["skill_match_score"].round(2)
    final_df.to_csv(output_file, index=False)


def main():
    """Run end-to-end resume screening workflow."""
    base_path = Path(__file__).resolve().parent
    resume_file = base_path / "resumes.csv"
    job_file = base_path / "job_description.txt"
    output_file = base_path / "screened_candidates.csv"

    try:
        resumes_df, job_description = load_data(str(resume_file), str(job_file))
        required_keywords, preferred_experience = extract_keywords(job_description)

        ranked_df = rank_candidates(resumes_df, required_keywords, preferred_experience)
        save_results(ranked_df, str(output_file))

        print("Top Candidates:")
        for _, row in ranked_df.head(5).iterrows():
            print(f"{int(row['rank'])}. {row['candidate_name']} - Score: {int(round(row['total_score']))}")

        print(f"\nScreening complete. Results saved to: {output_file}")

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
