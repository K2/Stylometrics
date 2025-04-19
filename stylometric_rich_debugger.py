import json
import time
import numpy as np
import argparse
from nltk.corpus import cmudict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.progress import track
import ollama

# Ensure NLTK's CMU Pronouncing Dictionary is downloaded
import nltk
nltk.download('cmudict')

# Initialize CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()
phoneme_set = sorted(set(p for v in cmu_dict.values() for p in v[0]))

# Function to convert text to phonemes
def text_to_phonemes(text):
    return [p for word in text.lower().split() if word in cmu_dict for p in cmu_dict[word][0]]

# Function to convert phonemes to a vector
def phoneme_vector(phonemes):
    vec = np.zeros(len(phoneme_set))
    for p in phonemes:
        if p in phoneme_set:
            vec[phoneme_set.index(p)] += 1
    return vec

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)

# Function to generate paraphrase using Ollama
def ollama_paraphrase(text, model):
    prompt = f"Paraphrase: {text}"
    response = ollama.chat(model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content'].strip()

# Function to generate harmonic or dissonant paraphrase pair
def generate_pair_ollama(base, model, resonance='harmonic'):
    if resonance == 'harmonic':
        prompt = f"Make this sentence smoother in cadence and sonics: '{base}'"
    else:
        prompt = f"Disrupt the sonic flow of this sentence: '{base}'"
    response = ollama.chat(model, messages=[{'role': 'user', 'content': prompt}])
    sentences = response['message']['content'].strip().split('\n')
    return sentences[0] if sentences else base

# Function to load datasets
def load_datasets(paths):
    corpus = []
    for path in paths:
        with open(path, 'r') as f:
            data = json.load(f)
            corpus.extend(data.get('corpus', []))
    return corpus

# Function to train classifier
def train_classifier(features, labels):
    clf = LogisticRegression()
    clf.fit(features, labels)
    return clf

# Function to create Rich table
def create_table():
    table = Table(show_lines=True)
    table.add_column("Index", justify="right")
    table.add_column("Score", justify="center")
    table.add_column("Expected", justify="center")
    table.add_column("Predicted", justify="center")
    return table

# Function to create histogram panel
def create_histogram_panel(similarities, labels, bins=10):
    sim0 = [s for s, l in zip(similarities, labels) if l == 0]
    sim1 = [s for s, l in zip(similarities, labels) if l == 1]
    hist0, edges = np.histogram(sim0, bins=bins, range=(0, 1))
    hist1, _ = np.histogram(sim1, bins=bins, range=(0, 1))
    bar_lines = []
    for h0, h1, edge in zip(hist0, hist1, edges[:-1]):
        block = f"{edge:.2f}-{edge + (edges[1]-edges[0]):.2f}: "
        block += "[blue]" + "█" * h0 + "[/blue]"
        block += "[magenta]" + "█" * h1 + "[/magenta]"
        block += f"  ({h0}/{h1})"
        bar_lines.append(block)
    hist_panel = Panel("\n".join(bar_lines), title="Histogram [blue]0[/blue] vs [magenta]1[/magenta]", subtitle="Phoneme Similarity")
    return hist_panel

# Function to create log panel
def create_log_panel(log_messages):
    text = "\n".join(log_messages[-10:])
    return Panel(Text(text, style="white on black"), title="Runtime Logs", subtitle="Last 10 entries")

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model to use for paraphrasing")
    parser.add_argument("--n", type=int, default=10, help="Number of paraphrase pairs per class")
    args = parser.parse_args()

    # Load datasets
    dataset_paths = ["data/set1.json", "data/set2.json"]
    corpus = load_datasets(dataset_paths)

    # Initialize variables
    pairs = []
    labels = []
    similarities = []
    log_buffer = []

    # Create Rich components
    console = Console()
    table = create_table()
    panel = Panel(table, title="Phoneme Cosine Similarities")
    layout = Layout()
    layout.split_row(
        Layout(panel, name="upper", size=None),
        Layout(name="middle", size=None),
        Layout(name="log", size=None)
    )

    # Start Rich Live display
    with Live(layout, console=console, auto_refresh=True) as live:
        for idx, entry in enumerate(track(corpus[:args.n], description="Processing entries...")):
            
            base = entry['content']
            console.log(f"Processing entry {idx}: {base}")
            # Generate paraphrase pairs
            harmonic = generate_pair_ollama(base, model=args.model, resonance='harmonic')
            dissonant = generate_pair_ollama(base, model=args.model, resonance='dissonant')
            console.log(f"Harmonic: {harmonic}")
            console.log(f"Dissonant: {dissonant}")

            # Compute similarities
            base_phonemes = text_to_phonemes(base)
            harmonic_phonemes = text_to_phonemes(harmonic)
            dissonant_phonemes = text_to_phonemes(dissonant)
            console.log(f"base:{base_phonemes},  harmonic:{harmonic_phonemes}, dissonant:{dissonant_phonemes}")

            base_vec = phoneme_vector(base_phonemes)
            harmonic_vec = phoneme_vector(harmonic_phonemes)
            dissonant_vec = phoneme_vector(dissonant_phonemes)

            sim_harmonic = cosine_similarity(base_vec, harmonic_vec)
            sim_dissonant = cosine_similarity(base_vec, dissonant_vec)

            # Append data
            pairs.extend([(base, harmonic), (base, dissonant)])
            labels.extend([1, 0])
            similarities.extend([sim_harmonic, sim_dissonant])
 
            # Update Rich UI
            pred_h = 1 if sim_harmonic >= 0.5 else 0
            pred_d = 1 if sim_dissonant >= 0.5 else 0
            similarities.extend([sim_harmonic, sim_dissonant])

            # Update Rich UI
            pred_h = 1 if sim_harmonic >= 0.5 else 0
            pred_d = 1 if sim_dissonant >= 0.5 else 0

            table.add_row(str(2*idx), f"{sim_harmonic:.3f}", "1", f"[{'green' if pred_h == 1 else 'red'}]{pred_h}[/]")
            table.add_row(str(2*idx+1), f"{sim_dissonant:.3f}", "0", f"[{'green' if pred_d == 0 else 'red'}]{pred_d}[/]")

            hist_panel = create_histogram_panel(similarities, labels)
            log_buffer.append(f"[{idx}] CosSim H={sim_harmonic:.3f}, D={sim_dissonant:.3f}, Acc={(pred_h==1 and pred_d==0):.0f}")
            log_panel = create_log_panel(log_buffer)

            layout["middle"].update(hist_panel)
            layout["log"].update(log_panel)
            #live.console.print(f"Processed #{idx}: H={sim_harmonic:.3f} D={sim_dissonant:.3f}")

    # End of live session
    features = [[cosine_similarity(
        phoneme_vector(text_to_phonemes(p[0])),
        phoneme_vector(text_to_phonemes(p[1]))
    )] for p in pairs]

    clf = train_classifier(features, labels)
    preds = clf.predict(features)
    print("\nFinal Classification Report:")
    print(classification_report(labels, preds, zero_division=1))



main()