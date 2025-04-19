
import streamlit as st
import pandas as pd
import numpy as np
import sys
from nltk.corpus import cmudict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Phoneme tools
cmu_dict = cmudict.dict()

def text_to_phonemes(text):
    return [p for word in text.lower().split() if word in cmu_dict for p in cmu_dict[word][0]]

def phoneme_vector(phonemes):
    phoneme_set = sorted(set(p for v in cmu_dict.values() for p in v[0]))
    vec = np.zeros(len(phoneme_set))
    for p in phonemes:
        if p in phoneme_set:
            vec[phoneme_set.index(p)] += 1
    return vec

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)

def train_classifier(features, labels):
    clf = LogisticRegression()
    clf.fit(features, labels)
    return clf

# --- Streamlit UI for secondary scrollable table output ---
def display_results_table(pairs, similarities, labels):
    data = []
    for idx, ((s1, s2), sim, label) in enumerate(zip(pairs, similarities, labels)):
        pred = 1 if sim >= 0.5 else 0
        correct = int(pred == label)
        data.append({
            "Pair Index": idx,
            "Similarity": f"{sim:.3f}",
            "Label": label,
            "Prediction": pred,
            "Correct": correct
        })
    df = pd.DataFrame(data)
    st.title("Phonetic Resonance Classification Results")
    st.dataframe(df, height=600)

# Main driver
def main():
    pairs = []
    similarities = []
    labels = []

    for idx in range(5):
        base = f"Example sentence {idx} to illustrate cadence."
        harmonic = base + " softly echoes the meaning."
        dissonant = base + " disrupted its style awkwardly."

        vec_base = phoneme_vector(text_to_phonemes(base))
        vec_harm = phoneme_vector(text_to_phonemes(harmonic))
        vec_diss = phoneme_vector(text_to_phonemes(dissonant))

        sim_harmonic = cosine_similarity(vec_base, vec_harm)
        sim_dissonant = cosine_similarity(vec_base, vec_diss)

        pairs.extend([(base, harmonic), (base, dissonant)])
        labels.extend([1, 0])
        similarities.extend([sim_harmonic, sim_dissonant])

    # Optional: display in Streamlit if run with --streamlit flag
    if '--streamlit' in sys.argv:
        display_results_table(pairs, similarities, labels)

    # Final classifier
    features = [[cosine_similarity(
        phoneme_vector(text_to_phonemes(p[0])),
        phoneme_vector(text_to_phonemes(p[1]))
    )] for p in pairs]

    clf = train_classifier(features, labels)
    preds = clf.predict(features)
    print("\nFinal Classification Report:")
    print(classification_report(labels, preds, zero_division=1))

if __name__ == "__main__":
    main()
