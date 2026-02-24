import os
import sys
import time

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.live import Live
from rich.markdown import Markdown

from continual_learning.config import load_config, save_config

console = Console()

# Global state
_model = None
_tokenizer = None
_dual_mlps = None
_ttt_engine = None
_checkpoint_manager = None
_config = None
_learning_history = []
_jitrl_mvp_engine = None
_jitrl_full_engine = None


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def format_status_panel(
    model_name: str,
    status: str,
    learned_count: int,
    vram_used: str = "N/A",
    vram_total: str = "N/A",
) -> str:
    return (
        f"[bold cyan]Model:[/bold cyan] {model_name}  |  "
        f"[bold cyan]Status:[/bold cyan] {status}  |  "
        f"[bold cyan]Learned:[/bold cyan] {learned_count} documents  |  "
        f"[bold cyan]VRAM:[/bold cyan] {vram_used} / {vram_total}"
    )


def print_header():
    status = "Ready" if _model is not None else "Not Loaded"
    model_name = _config["model"]["name"] if _config else "Qwen2.5-1.5B"
    learned = len(_learning_history)

    vram_used, vram_total = "N/A", "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            vram_used = f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"
            vram_total = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB"
    except Exception:
        pass

    status_text = format_status_panel(model_name, status, learned, vram_used, vram_total)
    console.print(Panel.fit(
        f"[bold cyan]Continual Learning SLM[/bold cyan]\n{status_text}",
        border_style="cyan",
    ))


def build_menu_choices() -> list:
    return [
        questionary.Choice("Chat with Model", value="chat"),
        questionary.Choice("Ask a Question", value="ask"),
        questionary.Separator("─── LEARNING ───────────────"),
        questionary.Choice("Learn from Document", value="learn_doc"),
        questionary.Choice("Learn from Directory", value="learn_dir"),
        questionary.Separator("─── TRAINING ───────────────"),
        questionary.Choice("Run Outer Loop (Meta-Learning)", value="outer_loop"),
        questionary.Separator("─── EVALUATION ─────────────"),
        questionary.Choice("Run Benchmarks", value="benchmark"),
        questionary.Choice("View Forgetting Metrics", value="forgetting"),
        questionary.Separator("─── MODEL ──────────────────"),
        questionary.Choice("Model Info", value="model_info"),
        questionary.Choice("Learning History", value="history"),
        questionary.Separator("─── CHECKPOINTS ────────────"),
        questionary.Choice("Save Checkpoint", value="save_cp"),
        questionary.Choice("Load Checkpoint", value="load_cp"),
        questionary.Choice("List Checkpoints", value="list_cp"),
        questionary.Separator("─── JITRL COMPARISON ───────"),
        questionary.Choice("JitRL MVP (Learn Doc)", value="jitrl_mvp"),
        questionary.Choice("JitRL Full (Learn Doc)", value="jitrl_full"),
        questionary.Choice("Compare All Engines", value="compare_engines"),
        questionary.Separator("─── SETTINGS ───────────────"),
        questionary.Choice("Configure", value="configure"),
        questionary.Separator("─── EXIT ───────────────────"),
        questionary.Choice("Exit", value="exit"),
    ]


def ensure_model_loaded():
    global _model, _tokenizer, _dual_mlps, _ttt_engine, _checkpoint_manager

    if _model is not None:
        return True

    console.print("[yellow]Loading model...[/yellow]")

    try:
        from continual_learning.model.modified_qwen import load_modified_model
        from continual_learning.model.dual_mlp import DualMLP
        from continual_learning.training.ttt_engine import TTTEngine
        from continual_learning.checkpointing.manager import CheckpointManager

        with console.status("[bold cyan]Downloading and modifying model..."):
            _model, _tokenizer = load_modified_model(
                model_name=_config["model"]["name"],
                device=_config["model"]["device"],
                start_layer=_config["model"]["modified_layers_start"],
                end_layer=_config["model"]["modified_layers_end"],
                alpha_initial=_config["alpha"]["initial"],
                tfidf_threshold=_config["tfidf_gate"]["threshold"],
            )

        _dual_mlps = []
        for layer in _model.model.layers:
            if isinstance(layer.mlp, DualMLP):
                _dual_mlps.append(layer.mlp)

        _ttt_engine = TTTEngine(
            model=_model,
            tokenizer=_tokenizer,
            dual_mlps=_dual_mlps,
            learning_rate=_config["ttt"]["learning_rate"],
            mini_batch_size=_config["ttt"]["mini_batch_size"],
            gradient_steps=_config["ttt"]["gradient_steps"],
            max_tokens=_config["ttt"]["max_tokens_per_document"],
        )

        _checkpoint_manager = CheckpointManager(
            checkpoint_dir=_config["checkpoint"]["dir"],
        )

        console.print("[green]Model loaded successfully![/green]")
        return True

    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return False


def handle_chat():
    if not ensure_model_loaded():
        return

    console.rule("[bold cyan]Chat Mode[/bold cyan]")
    console.print("[dim]Type 'quit' to return to menu[/dim]\n")

    import torch

    while True:
        query = Prompt.ask("[bold green]You[/bold green]")
        if query.lower() in ("quit", "exit", "q"):
            break

        inputs = _tokenizer(query, return_tensors="pt")
        device = next(_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with console.status("[cyan]Thinking...[/cyan]"):
            with torch.no_grad():
                outputs = _model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            response = _tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

        console.print(f"\n[bold cyan]Model[/bold cyan]: {response}\n")


def handle_ask():
    if not ensure_model_loaded():
        return

    import torch

    query = Prompt.ask("[bold green]Question[/bold green]")
    inputs = _tokenizer(query, return_tensors="pt")
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with console.status("[cyan]Thinking...[/cyan]"):
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        response = _tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    console.print(f"\n[bold cyan]Answer[/bold cyan]: {response}\n")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_learn_document():
    if not ensure_model_loaded():
        return

    file_path = Prompt.ask("[bold green]Document path[/bold green]")
    if not os.path.exists(file_path):
        console.print(f"[red]File not found: {file_path}[/red]")
        return

    with open(file_path) as f:
        text = f.read()

    console.print(f"[cyan]Document loaded: {len(text)} characters[/cyan]")

    def progress_callback(batch_idx, loss, tokens_so_far):
        console.print(
            f"  [dim]Batch {batch_idx}: loss={loss:.4f}, tokens={tokens_so_far}[/dim]"
        )

    console.rule("[bold cyan]Learning...[/bold cyan]")
    metrics = _ttt_engine.learn(text, callback=progress_callback)

    _learning_history.append({
        "file": file_path,
        "tokens": metrics["tokens_processed"],
        "final_loss": metrics["losses"][-1] if metrics["losses"] else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    console.print(f"\n[green]Learning complete![/green]")
    console.print(f"  Tokens processed: {metrics['tokens_processed']}")
    if metrics["losses"]:
        console.print(f"  Final loss: {metrics['losses'][-1]:.4f}")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_learn_directory():
    if not ensure_model_loaded():
        return

    dir_path = Prompt.ask("[bold green]Directory path[/bold green]")
    if not os.path.isdir(dir_path):
        console.print(f"[red]Directory not found: {dir_path}[/red]")
        return

    files = [f for f in os.listdir(dir_path) if f.endswith((".txt", ".md", ".jsonl"))]
    console.print(f"[cyan]Found {len(files)} documents[/cyan]")

    for i, filename in enumerate(files):
        filepath = os.path.join(dir_path, filename)
        console.print(f"\n[cyan]({i+1}/{len(files)}) Learning: {filename}[/cyan]")
        with open(filepath) as f:
            text = f.read()
        metrics = _ttt_engine.learn(text)
        _learning_history.append({
            "file": filepath,
            "tokens": metrics["tokens_processed"],
            "final_loss": metrics["losses"][-1] if metrics["losses"] else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        console.print(f"  [green]Done[/green] - tokens: {metrics['tokens_processed']}")

    console.print(f"\n[green]Finished learning from {len(files)} documents.[/green]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_benchmark():
    if not ensure_model_loaded():
        return

    from continual_learning.data.streaming_qa import load_squad_splits
    from continual_learning.evaluation.benchmarks import run_benchmark_suite

    console.print("[cyan]Loading SQuAD benchmark data...[/cyan]")
    _, holdout = load_squad_splits(learn_size=50, holdout_size=50, seed=42)

    with console.status("[bold cyan]Running benchmarks..."):
        results = run_benchmark_suite(_model, _tokenizer, holdout)

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Accuracy", f"{results['accuracy']:.4f}")
    table.add_row("Questions Evaluated", str(results["num_evaluated"]))
    if "forgetting_ratio" in results:
        table.add_row("Forgetting Ratio", f"{results['forgetting_ratio']:.4f}")

    console.print(table)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_model_info():
    if _model is None:
        console.print("[yellow]Model not loaded yet.[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
        return

    from continual_learning.model.dual_mlp import DualMLP

    table = Table(title="Model Architecture")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", _config["model"]["name"])
    table.add_row("Total Layers", str(len(_model.model.layers)))

    modified = sum(1 for l in _model.model.layers if isinstance(l.mlp, DualMLP))
    table.add_row("Modified Layers (DualMLP)", str(modified))
    table.add_row("Frozen Layers", str(len(_model.model.layers) - modified))

    total_params = sum(p.numel() for p in _model.parameters())
    trainable_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    table.add_row("Total Parameters", f"{total_params:,}")
    table.add_row("Trainable Parameters", f"{trainable_params:,}")
    table.add_row("Trainable %", f"{100 * trainable_params / total_params:.2f}%")

    if _dual_mlps:
        table.add_row("Alpha", f"{_dual_mlps[0].alpha:.4f}")
        calibrated = _dual_mlps[0].gate.idf_scores is not None
        table.add_row("TF-IDF Calibrated", str(calibrated))

    console.print(table)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_history():
    if not _learning_history:
        console.print("[yellow]No learning history yet.[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
        return

    table = Table(title="Learning History")
    table.add_column("#", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Tokens", style="green")
    table.add_column("Loss", style="yellow")
    table.add_column("Timestamp", style="dim")

    for i, entry in enumerate(_learning_history):
        table.add_row(
            str(i + 1),
            os.path.basename(entry["file"]),
            str(entry["tokens"]),
            f"{entry['final_loss']:.4f}" if entry["final_loss"] else "N/A",
            entry["timestamp"],
        )

    console.print(table)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_save_checkpoint():
    if not ensure_model_loaded():
        return

    name = Prompt.ask("[bold green]Checkpoint name[/bold green]")
    metadata = {
        "learning_history": _learning_history,
        "config": _config,
    }
    path = _checkpoint_manager.save(name, dual_mlps=_dual_mlps, metadata=metadata)
    console.print(f"[green]Checkpoint saved to: {path}[/green]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_load_checkpoint():
    if not ensure_model_loaded():
        return

    global _learning_history

    names = _checkpoint_manager.list()
    if not names:
        console.print("[yellow]No checkpoints found.[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
        return

    name = questionary.select(
        "Select checkpoint:",
        choices=names,
        use_arrow_keys=True,
    ).ask()

    if name:
        metadata = _checkpoint_manager.load(name, dual_mlps=_dual_mlps)
        if "learning_history" in metadata:
            _learning_history = metadata["learning_history"]
        console.print(f"[green]Checkpoint '{name}' loaded.[/green]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_list_checkpoints():
    if _checkpoint_manager is None:
        console.print("[yellow]Model not loaded yet.[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
        return

    names = _checkpoint_manager.list()
    if not names:
        console.print("[yellow]No checkpoints found.[/yellow]")
    else:
        table = Table(title="Checkpoints")
        table.add_column("Name", style="cyan")
        for name in names:
            table.add_row(name)
        console.print(table)

    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_configure():
    global _config

    console.rule("[bold cyan]Configuration[/bold cyan]")

    table = Table(title="Current Config")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    for section, values in _config.items():
        for key, val in values.items():
            table.add_row(f"{section}.{key}", str(val))
    console.print(table)

    if questionary.confirm("Edit configuration?").ask():
        setting = Prompt.ask("[bold green]Setting key (e.g. ttt.learning_rate)[/bold green]")
        parts = setting.split(".")
        if len(parts) == 2 and parts[0] in _config and parts[1] in _config[parts[0]]:
            current = _config[parts[0]][parts[1]]
            new_val = Prompt.ask(f"New value (current: {current})")
            if isinstance(current, float):
                _config[parts[0]][parts[1]] = float(new_val)
            elif isinstance(current, int):
                _config[parts[0]][parts[1]] = int(new_val)
            else:
                _config[parts[0]][parts[1]] = new_val
            save_config(_config)
            console.print("[green]Configuration saved.[/green]")
        else:
            console.print("[red]Invalid setting key.[/red]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_outer_loop():
    console.print("[yellow]Outer loop meta-learning is not yet implemented.[/yellow]")
    console.print("[dim]This will be added in a future milestone.[/dim]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_forgetting():
    console.print("[yellow]Detailed forgetting metrics view is not yet implemented.[/yellow]")
    console.print("[dim]Use 'Run Benchmarks' to see current forgetting ratio.[/dim]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")



def _ensure_jitrl_mvp():
    global _jitrl_mvp_engine
    if _jitrl_mvp_engine is not None:
        return True
    if not ensure_model_loaded():
        return False
    from continual_learning.jitrl.mvp.engine import JitRLMVPEngine
    _jitrl_mvp_engine = JitRLMVPEngine(
        model=_model, tokenizer=_tokenizer, top_k=3, bias_strength=2.0,
    )
    return True


def _ensure_jitrl_full():
    global _jitrl_full_engine
    if _jitrl_full_engine is not None:
        return True
    if not ensure_model_loaded():
        return False
    from continual_learning.jitrl.full.engine import JitRLFullEngine
    _jitrl_full_engine = JitRLFullEngine(
        model=_model, tokenizer=_tokenizer, modulation_temperature=0.5,
    )
    return True


def handle_jitrl_mvp():
    if not _ensure_jitrl_mvp():
        return

    file_path = Prompt.ask("[bold green]Document path[/bold green]")
    if not os.path.exists(file_path):
        console.print(f"[red]File not found: {file_path}[/red]")
        return

    with open(file_path) as f:
        text = f.read()

    console.print(f"[cyan]Indexing with JitRL MVP: {len(text)} chars[/cyan]")
    metrics = _jitrl_mvp_engine.learn(text)
    console.print(f"[green]Indexed! {metrics['chunks_added']} chunks, ~{metrics['tokens_processed']} tokens[/green]")

    query = Prompt.ask("[bold green]Ask a question[/bold green]")
    with console.status("[cyan]Generating (MVP)...[/cyan]"):
        response = _jitrl_mvp_engine.generate(query)
    console.print(f"\n[bold cyan]JitRL MVP[/bold cyan]: {response}\n")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_jitrl_full():
    if not _ensure_jitrl_full():
        return

    file_path = Prompt.ask("[bold green]Document path[/bold green]")
    if not os.path.exists(file_path):
        console.print(f"[red]File not found: {file_path}[/red]")
        return

    with open(file_path) as f:
        text = f.read()

    console.print(f"[cyan]Encoding with JitRL Full: {len(text)} chars[/cyan]")
    metrics = _jitrl_full_engine.learn(text)
    console.print(f"[green]Encoded! {metrics['tokens_processed']} tokens[/green]")

    query = Prompt.ask("[bold green]Ask a question[/bold green]")
    with console.status("[cyan]Generating (Full)...[/cyan]"):
        response = _jitrl_full_engine.generate(query)
    console.print(f"\n[bold cyan]JitRL Full[/bold cyan]: {response}\n")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


def handle_compare_engines():
    if not ensure_model_loaded():
        return
    _ensure_jitrl_mvp()
    _ensure_jitrl_full()

    from continual_learning.jitrl.comparison import ComparisonHarness

    file_path = Prompt.ask("[bold green]Document path to learn[/bold green]")
    if not os.path.exists(file_path):
        console.print(f"[red]File not found: {file_path}[/red]")
        return
    with open(file_path) as f:
        learn_text = f.read()

    console.print("[cyan]Enter QA pairs (empty question to stop):[/cyan]")
    qa_items = []
    while True:
        q = Prompt.ask("[green]Question[/green] (empty to stop)", default="")
        if not q:
            break
        a = Prompt.ask("[green]Expected answer[/green]")
        qa_items.append({"question": q, "answer": a})

    if not qa_items:
        console.print("[yellow]No QA items. Aborting comparison.[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
        return

    engines = {
        "JitRL MVP": _jitrl_mvp_engine,
        "JitRL Full": _jitrl_full_engine,
    }

    harness = ComparisonHarness(engines=engines)
    with console.status("[bold cyan]Running comparison..."):
        results = harness.run_comparison([learn_text], qa_items)

    table = Table(title="Engine Comparison")
    table.add_column("Engine", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Learn Time", style="yellow")
    table.add_column("Eval Time", style="yellow")
    table.add_column("Tokens Learned", style="dim")

    for name, r in results.items():
        table.add_row(
            name,
            f"{r['accuracy']:.2%}",
            f"{r['learn_time_s']:.2f}s",
            f"{r['eval_time_s']:.2f}s",
            str(r["tokens_learned"]),
        )
    console.print(table)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")


MENU_HANDLERS = {
    "chat": handle_chat,
    "ask": handle_ask,
    "learn_doc": handle_learn_document,
    "learn_dir": handle_learn_directory,
    "outer_loop": handle_outer_loop,
    "benchmark": handle_benchmark,
    "forgetting": handle_forgetting,
    "model_info": handle_model_info,
    "history": handle_history,
    "save_cp": handle_save_checkpoint,
    "load_cp": handle_load_checkpoint,
    "list_cp": handle_list_checkpoints,
    "configure": handle_configure,
    "jitrl_mvp": handle_jitrl_mvp,
    "jitrl_full": handle_jitrl_full,
    "compare_engines": handle_compare_engines,
}


def main_menu():
    while True:
        clear_screen()
        print_header()

        choice = questionary.select(
            "What would you like to do?",
            choices=build_menu_choices(),
            use_arrow_keys=True,
        ).ask()

        if not choice or choice == "exit":
            console.print("[dim]Goodbye![/dim]")
            sys.exit(0)

        handler = MENU_HANDLERS.get(choice)
        if handler:
            handler()
        else:
            console.print(f"[red]Unknown option: {choice}[/red]")


def main():
    global _config
    try:
        _config = load_config()
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
