
import torch
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import cmudict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Local chatpig T5 model for paraphrasing (replace with correct local path or model)
T5_MODEL = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"

cmu_dict = cmudict.dict()
phoneme_set = sorted(set(p for v in cmu_dict.values() for p in v[0]))

# Load local T5 paraphrasing model
tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).eval()

def text_to_phonemes(text):
    return [p for word in text.lower().split() if word in cmu_dict for p in cmu_dict[word][0]]

def phoneme_vector(phonemes):
    vec = np.zeros(len(phoneme_set))
    for p in phonemes:
        if p in phoneme_set:
            vec[phoneme_set.index(p)] += 1
    return vec

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)

def generate_pair_local(base, resonance='harmonic'):
    if resonance == 'harmonic':
        return base + " softly echoed the melody.", base + " whispered with grace."
    else:
        return base + " jerked across with noise.", base + " twisted in disruption."

def t5_paraphrase_local(text):
    inputs = tokenizer("paraphrase: " + text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def train_decoder(pairs, labels):
    features = []
    for s1, s2 in pairs:
        vec1, vec2 = phoneme_vector(text_to_phonemes(s1)), phoneme_vector(text_to_phonemes(s2))
        sim = cosine_similarity(vec1, vec2)
        features.append([sim])
    clf = LogisticRegression().fit(features, labels)
    return clf, features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="The sun glistened across the lake.", help="Base sentence")
    parser.add_argument("--n", type=int, default=10, help="Pairs per class")
    args = parser.parse_args()

    pairs, labels = [], []

    for _ in range(args.n):
        high, low = generate_pair_local(args.base, "harmonic"), generate_pair_local(args.base, "dissonant")
        pairs += [high, low]
        labels += [1, 0]

    clf, features = train_decoder(pairs, labels)
    paraphrased = [(t5_paraphrase_local(s1), t5_paraphrase_local(s2)) for s1, s2 in pairs]
    test_feats = []
    for s1, s2 in paraphrased:
        vec1, vec2 = phoneme_vector(text_to_phonemes(s1)), phoneme_vector(text_to_phonemes(s2))
        test_feats.append([cosine_similarity(vec1, vec2)])

    preds = clf.predict(test_feats)
    print("Classification Report on Paraphrased Pairs:")
    print(classification_report(labels, preds))

    with open("phonetic_pairs_local.json", "w") as f:
        json.dump([{"original": o, "paraphrased": p, "label": l}
                   for o, p, l in zip(pairs, paraphrased, labels)], f, indent=2)

    sims = [f[0] for f in test_feats]
    plt.hist(sims, bins=10, alpha=0.6, label="Paraphrased Similarity")
    plt.title("Phoneme Similarity Histogram (Local T5)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("resonance_plot_local.png")
    plt.show()

if __name__ == "__main__":
    main()
