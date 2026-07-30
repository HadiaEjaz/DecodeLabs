"""
Tech Stack Recommender — Project 3 Capstone (DecodeLabs)
----------------------------------------------------------
Implements a Content-Based Filtering recommendation engine using
TF-IDF feature weighting + Cosine Similarity, following the
4-step pipeline: Ingestion -> Scoring -> Sorting -> Filtering.

No external ML libraries used — pure Python logic, so every
step of the math is visible and auditable.
"""

import csv
import math
from collections import Counter


# ---------------------------------------------------------------
# STEP 0: Load the dataset ("Items" in our recommendation engine)
# ---------------------------------------------------------------
def load_job_roles(filepath):
    """Reads raw_skills.csv into {job_role: [skill1, skill2, ...]}"""
    roles = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row["job_role"].strip()
            skills = [s.strip().lower() for s in row["skills"].split(";")]
            roles[role] = skills
    return roles


# ---------------------------------------------------------------
# STEP 1 (Vector Mapping): Build the shared vocabulary space
# ---------------------------------------------------------------
def build_vocabulary(roles):
    vocab = set()
    for skills in roles.values():
        vocab.update(skills)
    return sorted(vocab)


# ---------------------------------------------------------------
# TF-IDF WEIGHTING
# ---------------------------------------------------------------
def compute_idf(roles, vocab):
    """IDF = log(total_docs / docs_containing_term)"""
    total_docs = len(roles)
    idf = {}
    for term in vocab:
        docs_with_term = sum(1 for skills in roles.values() if term in skills)
        # +1 smoothing avoids division by zero / log(0)
        idf[term] = math.log(total_docs / (1 + docs_with_term)) + 1
    return idf


def compute_tf(skills):
    """TF = count(term in doc) / total_terms_in_doc"""
    counts = Counter(skills)
    total = len(skills)
    return {term: count / total for term, count in counts.items()}


def build_tfidf_vector(skills, vocab, idf):
    """Builds a full-length weighted vector over the shared vocabulary."""
    tf = compute_tf(skills)
    return {term: tf.get(term, 0.0) * idf[term] for term in vocab}


# ---------------------------------------------------------------
# STEP 2 (Scoring): Cosine Similarity — the industry standard
# ---------------------------------------------------------------
def cosine_similarity(vec_a, vec_b):
    dot = sum(vec_a[t] * vec_b[t] for t in vec_a)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------
# COLD START HANDLING
# ---------------------------------------------------------------
def is_cold_start(user_skills, vocab):
    """True if none of the user's skills exist in our vocabulary."""
    return not any(skill.lower() in vocab for skill in user_skills)


def trending_fallback(roles, top_n=3):
    """Fallback: recommend the roles with the broadest (most in-demand) skillsets."""
    ranked = sorted(roles.items(), key=lambda kv: len(kv[1]), reverse=True)
    return [(role, None) for role, _ in ranked[:top_n]]


# ---------------------------------------------------------------
# STEP 1+2+3+4 COMBINED: The full recommendation pipeline
# ---------------------------------------------------------------
def recommend(user_skills, roles, vocab, idf, top_n=3):
    # --- Step 1: Ingestion ---
    if len(user_skills) < 3:
        raise ValueError("Please provide at least 3 skills for accurate matching.")

    if is_cold_start(user_skills, vocab):
        print("⚠️  Cold Start detected: none of your skills matched our vocabulary.")
        print("   Falling back to trending/popular roles.\n")
        return trending_fallback(roles, top_n)

    user_vector = build_tfidf_vector(
        [s.lower() for s in user_skills], vocab, idf
    )

    # --- Step 2: Scoring ---
    scored = []
    for role, skills in roles.items():
        role_vector = build_tfidf_vector(skills, vocab, idf)
        score = cosine_similarity(user_vector, role_vector)
        scored.append((role, score))

    # --- Step 3: Sorting ---
    scored.sort(key=lambda x: x[1], reverse=True)

    # --- Step 4: Filtering (Top-N) ---
    return scored[:top_n]


# ---------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------
if __name__ == "__main__":
    roles = load_job_roles("raw_skills.csv")
    vocab = build_vocabulary(roles)
    idf = compute_idf(roles, vocab)

    test_inputs = [
        ["Python", "Cloud", "Automation"],
        ["JavaScript", "React", "CSS"],
        ["Blockchain", "Solidity", "Web3"],   # cold start example
    ]

    for user_skills in test_inputs:
        print(f"User input: {user_skills}")
        results = recommend(user_skills, roles, vocab, idf, top_n=3)
        for rank, (role, score) in enumerate(results, start=1):
            score_display = f"{score:.3f}" if score is not None else "N/A (fallback)"
            print(f"  {rank}. {role:<22} score: {score_display}")
        print("-" * 50)
