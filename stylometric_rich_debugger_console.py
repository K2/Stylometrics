
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

def simulate_debug_view(similarities, labels):
    layout = Layout()
    log_buffer = []

    with Live(layout, refresh_per_second=3, screen=False) as live:
        for i in track(range(len(similarities)), description="Streaming decoder scores..."):
            time.sleep(0.15)
            partial_sims = similarities[:i+1]
            partial_labels = labels[:i+1]

            table, preds = render_similarity_table(partial_sims, partial_labels)
            hist_panel = render_histogram(partial_sims, partial_labels)
            correct = sum(p == l for p, l in zip(preds, partial_labels))
            acc = correct / len(preds)
            log_buffer.append(f"[{i}] accuracy={acc:.2%} | harmonic={partial_labels[-1]==1}")

            # print externally to scrollable terminal history above the layout
            live.console.print(f"[{i}] CosSim={partial_sims[-1]:.3f}, Label={partial_labels[-1]}, Pred={preds[-1]}, Acc={acc:.2%}")

            layout.split_column(
                Layout(table, name="upper", size=18),
                Layout(hist_panel, name="middle", size=12),
                Layout(render_logs(log_buffer), name="log", size=10)
            )
            live.update(layout)

def example_usage():
    np.random.seed(42)
    sims = np.clip(np.random.normal(0.6, 0.1, 50), 0, 1)
    labels = [1 if s > 0.55 else 0 for s in sims]
    simulate_debug_view(sims.tolist(), labels)

if __name__ == "__main__":
    example_usage()
