import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import cmudict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import ollama

cmu_dict = cmudict.dict()
phoneme_set = sorted(set(p for v in cmu_dict.values() for p in v[0]))
MODEL = ""
def text_to_phonemes(texts):
    return [p for text in texts for word in text.lower().split() if word in cmu_dict for p in cmu_dict[word][0]]

def phoneme_vector(phonemes):
    vec = np.zeros(len(phoneme_set))
    for p in phonemes:
        if p in phoneme_set:
            vec[phoneme_set.index(p)] += 1
    return vec

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)

def ollama_paraphrase(text, model=MODEL):
    prompt = f"Paraphrase: {text}"
    response = ollama.chat(model, messages=[{ 'role': 'user', 'content': prompt }])
    return response['message']['content'].strip()

def generate_pair_ollama(base, model=MODEL, resonance='harmonic'):
    if resonance == 'harmonic':
        prompt = f"Make this sentence smoother in cadence and sonics: '{base}'"
    else:
        prompt = f"Disrupt the sonic flow of this sentence: '{base}'"
    response = ollama.chat(model, messages=[{ 'role': 'user', 'content': prompt }])
    sentences = response['message']['content'].strip().split('\n')
    return sentences[0], sentences[1] if len(sentences) > 1 else sentences[0]

def train_decoder(pairs, labels):
    """
    Transform pairs of sentences into phonetic similarity features and train a classifier.
    
    Args:
        pairs: List of sentence pairs (tuples of two sentences)
        labels: Corresponding labels for each pair
    """
    # Convert each pair to a feature using cosine similarity of phoneme vectors
    features = list(map(
        lambda pair: [cosine_similarity(
            phoneme_vector(text_to_phonemes(pair[0])), 
            phoneme_vector(text_to_phonemes(pair[1]))
        )],
        zip(pairs[::2], pairs[1::2])  # Create pairs from adjacent elements
    ))
    
    clf = LogisticRegression().fit(features, labels)
    return clf, features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, help="Model to use for paraphrasing", required=True)
    parser.add_argument("--generate", action='store_true', help="Generate paraphrases")
    parser.add_argument("--resonance", choices=['harmonic', 'dissonant'], default='harmonic', help="Type of resonance for generation")
    parser.add_argument("--base-gen", action='store_true',  help="generate base sentence's for paraphrasing", default=False)
    parser.add_argument("--base", default="The sun glistened across the lake.", help="Base sentence")
    parser.add_argument("--n", type=int, default=10, help="Pairs per class")
    args = parser.parse_args()

    pairs, labels = [], []
    base = ""

    # if args.base:
    #     pairs.append(args.base)
    #     labels.append(1)
    # else:    
    #     pairs.append(args.base)
    #     labels.append(0)

    # if args.base_gen:
    #     for _ in range(args.n):
    #         pairs.append(args.base)
    #         labels.append(1)

    if args.generate:
        for _ in range(args.n):
            high, low = generate_pair_ollama(args.base, model=args.model, resonance=args.resonance), generate_pair_ollama(args.base, model=args.model, resonance="dissonant")
            pairs += [high, low]
            labels += [1, 0]

    clf, features = train_decoder(pairs, labels)
    paraphrased = [(ollama_paraphrase(s1, model=args.model), ollama_paraphrase(s2, model=args.model)) for s1, s2 in pairs]
    test_feats = []
    for s1, s2 in paraphrased:
        vec1, vec2 = phoneme_vector(text_to_phonemes(s1)), phoneme_vector(text_to_phonemes(s2))
        test_feats.append([cosine_similarity(vec1, vec2)])

    preds = clf.predict(test_feats)
    print("Classification Report on Ollama Paraphrased Pairs:")
    print(classification_report(labels, preds))

    with open("phonetic_pairs_ollama.json", "w") as f:
        json.dump([{"original": o, "paraphrased": p, "label": l}
                   for o, p, l in zip(pairs, paraphrased, labels)], f, indent=2)

    sims = [f[0] for f in test_feats]
    plt.hist(sims, bins=10, alpha=0.6, label="Ollama Paraphrased Similarity")
    plt.title("Phoneme Similarity Histogram (Ollama)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("resonance_plot_ollama.png")
    plt.show()

if __name__ == "__main__":
    main()
