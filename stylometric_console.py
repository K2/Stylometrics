# stylometric_console.py
"""
Stylometric Console: Unified Rich Console Monitor for Training, Finetuning, and Testing Phases

- Generic, extensible monitoring for all phases (train, finetune, test, eval, custom)
- Pluggable data streams (similarities, losses, metrics, logs, etc.)
- Phase-specific and phase-agnostic panels
- Example usage for dataset-driven training and testing

See stylometric_console.ApiNotes.md for design and extension details.
"""
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import track
import numpy as np
import time

console = Console()

def render_similarity_table(similarities, labels, threshold=0.5):
    table = Table(title="Phoneme Cosine Similarities", show_lines=True)
    table.add_column("Index", justify="right")
    table.add_column("Score", justify="center")
    table.add_column("Expected", justify="center")
    table.add_column("Predicted", justify="center")

    preds = []
    for i, (score, true_label) in enumerate(zip(similarities, labels)):
        pred = 1 if score >= threshold else 0
        preds.append(pred)
        color = "green" if pred == true_label else "red"
        label_text = Text(f"{pred}", style=color)
        table.add_row(str(i), f"{score:.3f}", str(true_label), label_text)

    return table, preds

def render_histogram(similarities, labels, bins=10):
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

def render_logs(log_messages):
    text = "\n".join(log_messages[-10:])
    return Panel(Text(text, style="white on black"), title="Runtime Logs", subtitle="Last 10 entries")

def render_loss_curve(losses):
    # Simple ASCII sparkline for loss curve
    if not losses:
        return Panel("No loss data.", title="Loss Curve")
    min_loss, max_loss = min(losses), max(losses)
    scaled = [int(10 * (l - min_loss) / (max_loss - min_loss + 1e-8)) for l in losses]
    spark = ''.join('▁▂▃▄▅▆▇█'[min(s, 7)] for s in scaled)
    return Panel(f"{spark}\nLast: {losses[-1]:.4f}", title="Loss Curve")

def run_console_monitor(phase, data_stream, **kwargs):
    """
    phase: str, e.g. 'train', 'finetune', 'test', 'eval', 'custom'
    data_stream: iterable of dicts with keys depending on phase
        For 'train': expects dict with 'similarities', 'labels', 'loss', 'log'
        For 'test': expects dict with 'similarities', 'labels', 'log'
    """
    layout = Layout()
    log_buffer = []
    loss_buffer = []

    with Live(layout, refresh_per_second=3, screen=False) as live:
        for i, data in enumerate(data_stream):
            similarities = data.get('similarities', [])
            labels = data.get('labels', [])
            loss = data.get('loss')
            log = data.get('log', "")
            if log:
                log_buffer.append(log)
            if loss is not None:
                loss_buffer.append(loss)

            table, preds = render_similarity_table(similarities, labels)
            hist_panel = render_histogram(similarities, labels)
            correct = sum(p == l for p, l in zip(preds, labels))
            acc = correct / len(preds) if preds else 0.0

            # Phase-specific panels
            panels = [Layout(table, name="upper", size=18),
                      Layout(hist_panel, name="middle", size=12),
                      Layout(render_logs(log_buffer), name="log", size=10)]
            if phase in ("train", "finetune"):
                panels.append(Layout(render_loss_curve(loss_buffer), name="loss", size=6))

            layout.split_column(*panels)
            live.console.print(f"[{i}] Acc={acc:.2%} | Loss={loss if loss is not None else '-'}")
            live.update(layout)
            time.sleep(kwargs.get('sleep', 0.15))

# Example usage for test and train phases

# --- Real Data Example: CAPT Dataset ---
import json
import os
from phonetic_helpers import text_to_phonemes, phoneme_vec, cosine

def example_realdata_usage():
    """
    Streams real CAPT dataset entries through the console monitor.
    Computes phoneme similarity between correct and distractor answers for each question.
    """
    capt_path = os.path.join(os.path.dirname(__file__), "data", "capt_test.json")
    with open(capt_path, "r", encoding="utf-8") as f:
        capt_data = json.load(f)
    entries = list(capt_data.values())

    def capt_stream():
        for i, entry in enumerate(entries):
            # For demonstration, compare phoneme vectors of correct vs. one distractor answer
            correct_ans = entry.get(entry["answer"], "")
            distractor_key = next(k for k in "ABCD" if k != entry["answer"] and entry.get(k))
            distractor_ans = entry.get(distractor_key, "")
            # Use first word of each answer for phoneme comparison (or all words joined)
            correct_phonemes = text_to_phonemes(correct_ans)
            distractor_phonemes = text_to_phonemes(distractor_ans)
            if correct_phonemes and distractor_phonemes:
                sim = cosine(phoneme_vec(correct_phonemes), phoneme_vec(distractor_phonemes))
            else:
                sim = 0.0
            # Label: 1 for correct, 0 for distractor
            similarities = [sim]
            labels = [1]
            yield {
                'similarities': similarities,
                'labels': labels,
                'log': f"[capt {i}] Q: {entry['question']} | Sim={sim:.3f}"
            }

    print("--- CAPT Dataset Phase (real data) ---")
    run_console_monitor('test', capt_stream(), sleep=0.3)


# --- Real Data Example: Concept Dataset ---
def example_conceptdata_usage():
    """
    Streams real Concept dataset entries through the console monitor.
    Computes phoneme or string similarity between correct and distractor answers for each question.
    """
    concept_path = os.path.join(os.path.dirname(__file__), "data", "concept_test.json")
    with open(concept_path, "r", encoding="utf-8") as f:
        concept_data = json.load(f)
    entries = list(concept_data.values())

    def concept_stream():
        for i, entry in enumerate(entries):
            correct_ans = entry.get(entry["answer"], "")
            distractor_key = next(k for k in "ABCD" if k != entry["answer"] and entry.get(k))
            distractor_ans = entry.get(distractor_key, "")
            # Try phoneme similarity, fallback to string similarity
            correct_phonemes = text_to_phonemes(correct_ans)
            distractor_phonemes = text_to_phonemes(distractor_ans)
            if correct_phonemes and distractor_phonemes:
                sim = cosine(phoneme_vec(correct_phonemes), phoneme_vec(distractor_phonemes))
            else:
                # Fallback: normalized Levenshtein similarity (or 0 if not available)
                try:
                    import difflib
                    sim = difflib.SequenceMatcher(None, correct_ans, distractor_ans).ratio()
                except Exception:
                    sim = 0.0
            similarities = [sim]
            labels = [1]
            yield {
                'similarities': similarities,
                'labels': labels,
                'log': f"[concept {i}] Q: {entry['question']} | Sim={sim:.3f}"
            }

    print("--- Concept Dataset Phase (real data) ---")
    run_console_monitor('test', concept_stream(), sleep=0.3)

if __name__ == "__main__":
    # example_usage()  # Old synthetic example
    example_realdata_usage()
    example_conceptdata_usage()
