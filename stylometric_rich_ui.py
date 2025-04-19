
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import track
import time
import numpy as np

console = Console()

def render_similarity_table(similarities, threshold=0.5):
    table = Table(title="Phoneme Cosine Similarities", show_lines=True)
    table.add_column("Index", justify="right")
    table.add_column("Score", justify="center")
    table.add_column("Signal", justify="center")

    for i, score in enumerate(similarities):
        label = "[green]Harmonic[/green]" if score >= threshold else "[red]Dissonant[/red]"
        table.add_row(str(i), f"{score:.3f}", label)

    return table

def render_histogram(similarities, bins=10):
    hist, edges = np.histogram(similarities, bins=bins)
    max_count = max(hist)
    layout = Layout()

    bars = ""
    for count, edge in zip(hist, edges[:-1]):
        height = int((count / max_count) * 10)
        bars += f"{edge:.2f}-{edge + (edges[1] - edges[0]):.2f}: " + "█" * height + f" ({count})\n"

    return Panel(bars.strip(), title="Real-Time Histogram", subtitle="Phoneme Cosine Distribution")

def simulate_stream(similarities):
    with Live(console=console, refresh_per_second=2) as live:
        for i in track(range(len(similarities)), description="Streaming similarity vectors..."):
            time.sleep(0.2)
            partial = similarities[:i + 1]
            table = render_similarity_table(partial)
            hist_panel = render_histogram(partial)
            live.update(Layout(name="main", renderable=table).split_row(Layout(hist_panel)))

def example_usage():
    similarities = np.clip(np.random.normal(0.6, 0.1, 30), 0, 1)
    simulate_stream(similarities.tolist())

if __name__ == "__main__":
    example_usage()
