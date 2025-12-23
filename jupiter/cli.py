"""
Jupiter CLI - Command line interface for the framework.

Commands:
    - start: Start training pipeline
    - check: Verify configuration
    - swarm: Run MoE-R swarm inference
    - swarm-train: Train expert swarm
    - model-info: Show model architecture info
"""

import asyncio
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from jupiter.config import JupiterConfig
from jupiter.training.model import ModelConfig

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    Jupiter - Distributed training framework for expert language models.

    Train 1B parameter models on Mac clusters and/or NVIDIA GPUs.
    Supports MoE-R (Mixture of Real Experts) for collaborative inference.
    """
    pass


@main.command()
@click.option("--domain", "-d", required=True, help="Nombre del dominio a usar")
@click.option("--config", "-c", default="config", help="Directorio de configuración")
@click.option("--mode", "-m", default="auto", type=click.Choice(["auto", "interactive"]))
@click.option("--epochs", "-e", default=100, help="Número máximo de épocas")
@click.option("--no-collect", is_flag=True, help="No recolectar datos reales")
@click.option("--no-generate", is_flag=True, help="No generar datos sintéticos")
def start(domain: str, config: str, mode: str, epochs: int, no_collect: bool, no_generate: bool):
    """
    Inicia el pipeline de training.

    Ejemplo:
        jupiter start --domain unreal_engine
    """
    console.print(Panel.fit(
        "[bold blue]Jupiter[/bold blue] - Training Pipeline",
        subtitle=f"Dominio: {domain}"
    ))

    try:
        # Cargar configuración
        config_path = Path(config)
        jupiter_config = JupiterConfig.from_yaml(config_path, domain)

        # Importar aquí para evitar imports lentos
        from jupiter.orchestrator import Orchestrator

        # Crear orchestrador
        orchestrator = Orchestrator(jupiter_config)

        # Ejecutar
        asyncio.run(orchestrator.run(
            mode=mode,
            max_epochs=epochs,
            collect_data=not no_collect,
            generate_data=not no_generate,
        ))

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrumpido por el usuario[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@main.command()
@click.option("--config", "-c", default="config", help="Directorio de configuración")
def check(config: str):
    """
    Verifica la configuración y el cluster.

    Ejemplo:
        jupiter check
    """
    console.print("[bold]Verificando configuración...[/bold]\n")

    config_path = Path(config)

    # Verificar archivos de configuración
    cluster_file = config_path / "cluster.yaml"
    domains_dir = config_path / "domains"

    table = Table(title="Archivos de Configuración")
    table.add_column("Archivo", style="cyan")
    table.add_column("Estado", style="green")

    if cluster_file.exists():
        table.add_row("cluster.yaml", "✓ Encontrado")
    else:
        table.add_row("cluster.yaml", "[red]✗ No encontrado[/red]")

    if domains_dir.exists():
        domains = list(domains_dir.glob("*.yaml"))
        domains = [d for d in domains if not d.name.startswith("_")]
        table.add_row(
            "Dominios",
            f"✓ {len(domains)} encontrados: {', '.join(d.stem for d in domains)}"
        )
    else:
        table.add_row("Dominios", "[red]✗ Directorio no encontrado[/red]")

    console.print(table)

    # Verificar cluster si existe
    if cluster_file.exists():
        console.print("\n[bold]Verificando cluster...[/bold]")

        from jupiter.config import ClusterConfig
        import yaml

        with open(cluster_file) as f:
            cluster_data = yaml.safe_load(f)
            cluster = ClusterConfig.from_dict(cluster_data)

        errors = cluster.validate()
        if errors:
            for error in errors:
                console.print(f"[red]✗[/red] {error}")
        else:
            console.print("[green]✓[/green] Configuración del cluster válida")

            cluster_table = Table(title="Nodos del Cluster")
            cluster_table.add_column("Nombre")
            cluster_table.add_column("Host")
            cluster_table.add_column("Tipo")
            cluster_table.add_column("RAM")
            cluster_table.add_column("Rol")

            for node in cluster.nodes:
                cluster_table.add_row(
                    node.display_name,
                    node.host,
                    node.device_type.value,
                    f"{node.memory_gb}GB",
                    node.role.value,
                )

            console.print(cluster_table)
            console.print(f"\nMemoria total: [bold]{cluster.total_memory_gb}GB[/bold]")


@main.command()
@click.argument("name")
@click.option("--output", "-o", default="config/domains", help="Directorio de salida")
def new_domain(name: str, output: str):
    """
    Crea un nuevo dominio desde el template.

    Ejemplo:
        jupiter new-domain python_expert
    """
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    template_path = output_path / "_template.yaml"
    new_path = output_path / f"{name}.yaml"

    if new_path.exists():
        console.print(f"[red]Error:[/red] El dominio '{name}' ya existe")
        raise SystemExit(1)

    if template_path.exists():
        import shutil
        shutil.copy(template_path, new_path)
        console.print(f"[green]✓[/green] Dominio creado: {new_path}")
        console.print(f"  Edita el archivo para configurar tu dominio")
    else:
        console.print(f"[red]Error:[/red] Template no encontrado en {template_path}")
        raise SystemExit(1)


@main.command()
@click.option("--preset", "-p", default="1b", type=click.Choice(["125m", "350m", "500m", "1b", "3b"]))
def model_info(preset: str):
    """
    Muestra información sobre las arquitecturas de modelo.

    Ejemplo:
        jupiter model-info --preset 1b
    """
    config = ModelConfig.from_preset(preset)

    table = Table(title=f"Arquitectura: {config.name}")
    table.add_column("Parámetro", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Parámetros", config.num_parameters_str)
    table.add_row("Capas", str(config.num_hidden_layers))
    table.add_row("Hidden size", str(config.hidden_size))
    table.add_row("Intermediate size", str(config.intermediate_size))
    table.add_row("Attention heads", str(config.num_attention_heads))
    table.add_row("KV heads", str(config.num_key_value_heads))
    table.add_row("Max seq length", str(config.max_position_embeddings))
    table.add_row("Vocab size", str(config.vocab_size))
    table.add_row("Memoria estimada (training)", f"{config.estimated_memory_gb:.1f}GB")

    console.print(table)

    # Mostrar todas las arquitecturas
    console.print("\n[bold]Arquitecturas disponibles:[/bold]")
    all_presets = ["125m", "350m", "500m", "1b", "3b"]

    for p in all_presets:
        c = ModelConfig.from_preset(p)
        marker = "→" if p == preset else " "
        console.print(f"  {marker} {p}: {c.num_parameters_str} parámetros, ~{c.estimated_memory_gb:.0f}GB RAM")


@main.command()
@click.option("--config", "-c", default="config", help="Directorio de configuración")
def status(config: str):
    """
    Muestra el estado actual del pipeline.
    """
    import json

    state_file = Path("pipeline_state.json")

    if not state_file.exists():
        console.print("[yellow]No hay pipeline en ejecución[/yellow]")
        return

    with open(state_file) as f:
        state = json.load(f)

    console.print(Panel.fit(
        f"[bold]Estado del Pipeline[/bold]\n"
        f"Fase: {state.get('phase', 'unknown')}\n"
        f"Época: {state.get('current_epoch', 0)}\n"
        f"Documentos: {state.get('total_documents', 0)}\n"
        f"Sintéticos: {state.get('total_synthetic', 0)}\n"
        f"Mejor score: {state.get('best_eval_score', 0):.4f}\n"
        f"Versión generador: {state.get('generator_version', 0)}"
    ))

    if "stats" in state:
        stats = state["stats"]
        table = Table(title="Estadísticas")
        table.add_column("Métrica")
        table.add_column("Valor")

        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)


@main.command()
@click.argument("checkpoint_path")
@click.option("--format", "-f", default="mlx", type=click.Choice(["mlx", "hf", "gguf"]))
@click.option("--output", "-o", help="Directorio de salida")
def export(checkpoint_path: str, format: str, output: Optional[str]):
    """
    Exporta un modelo entrenado.

    Ejemplo:
        jupiter export checkpoints/best --format hf
    """
    checkpoint = Path(checkpoint_path)

    if not checkpoint.exists():
        console.print(f"[red]Error:[/red] Checkpoint no encontrado: {checkpoint}")
        raise SystemExit(1)

    output_path = Path(output) if output else checkpoint.parent / f"export_{format}"

    console.print(f"Exportando {checkpoint} a formato {format}...")

    # TODO: Implementar exportación
    console.print(f"[yellow]Exportación a {format} no implementada aún[/yellow]")


# =============================================================================
# SWARM COMMANDS (MoE-R)
# =============================================================================

@main.group()
def swarm():
    """
    MoE-R Swarm commands - Multiple expert collaboration.

    Use specialized expert models that work together to solve complex tasks.
    """
    pass


@swarm.command("chat")
@click.option("--experts", "-e", default="config/experts", help="Experts configuration directory")
@click.option("--config", "-c", default=None, help="Swarm configuration file")
@click.option("--load", "-l", multiple=True, help="Specific experts to load (can use multiple times)")
def swarm_chat(experts: str, config: Optional[str], load: tuple):
    """
    Start interactive chat with the expert swarm.

    Example:
        jupiter swarm chat --experts config/experts/python
        jupiter swarm chat -l python-core -l python-backend
    """
    from jupiter.swarm import Swarm, SwarmConfig

    console.print(Panel.fit(
        "[bold blue]Jupiter Swarm[/bold blue] - MoE-R Chat",
        subtitle="Mixture of Real Experts"
    ))

    try:
        # Load swarm config
        if config:
            swarm_config = SwarmConfig.from_yaml(config)
        else:
            swarm_config = SwarmConfig(experts_dir=experts)

        # Create swarm
        swarm_instance = Swarm(swarm_config)

        async def run_chat():
            await swarm_instance.initialize()

            # Load specific experts if requested
            if load:
                for expert_name in load:
                    try:
                        await swarm_instance.load_expert(expert_name)
                    except Exception as e:
                        console.print(f"[yellow]Warning:[/yellow] Could not load {expert_name}: {e}")

            # Run chat
            async for _ in swarm_instance.chat():
                pass

            await swarm_instance.shutdown()

        asyncio.run(run_chat())

    except KeyboardInterrupt:
        console.print("\n[yellow]Chat ended[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@swarm.command("list")
@click.option("--experts", "-e", default="config/experts", help="Experts configuration directory")
def swarm_list(experts: str):
    """
    List available expert configurations.

    Example:
        jupiter swarm list
    """
    experts_path = Path(experts)

    if not experts_path.exists():
        console.print(f"[red]Error:[/red] Experts directory not found: {experts_path}")
        return

    table = Table(title="Available Experts")
    table.add_column("Name", style="cyan")
    table.add_column("Domain", style="green")
    table.add_column("Description")
    table.add_column("Keywords", style="dim")

    # Find all expert configs recursively
    for config_file in experts_path.rglob("*.yaml"):
        if config_file.name.startswith("_"):
            continue

        try:
            from jupiter.swarm import ExpertConfig
            config = ExpertConfig.from_yaml(str(config_file))

            keywords = ", ".join(config.keywords[:5])
            if len(config.keywords) > 5:
                keywords += f" (+{len(config.keywords) - 5})"

            table.add_row(
                config.name,
                config.domain,
                config.description[:50] + "..." if len(config.description) > 50 else config.description,
                keywords,
            )
        except Exception as e:
            table.add_row(config_file.stem, "[red]Error[/red]", str(e)[:50], "")

    console.print(table)


@swarm.command("query")
@click.argument("query")
@click.option("--experts", "-e", default="config/experts", help="Experts configuration directory")
@click.option("--load", "-l", multiple=True, help="Specific experts to load")
@click.option("--top-k", "-k", default=3, help="Number of experts to use")
def swarm_query(query: str, experts: str, load: tuple, top_k: int):
    """
    Send a single query to the swarm.

    Example:
        jupiter swarm query "How do I create a FastAPI endpoint?"
        jupiter swarm query "Explain React hooks" -l react-frontend
    """
    from jupiter.swarm import Swarm, SwarmConfig, RouterConfig

    console.print(f"[bold]Query:[/bold] {query}\n")

    try:
        router_config = RouterConfig(top_k=top_k)
        swarm_config = SwarmConfig(experts_dir=experts, router=router_config)
        swarm_instance = Swarm(swarm_config)

        async def run_query():
            await swarm_instance.initialize()

            # Load specific experts
            if load:
                for expert_name in load:
                    try:
                        await swarm_instance.load_expert(expert_name)
                    except Exception as e:
                        console.print(f"[yellow]Warning:[/yellow] {e}")

            with console.status("[bold green]Experts thinking..."):
                response = await swarm_instance.query(query)

            console.print(Panel(
                response.content,
                title="[bold green]Response[/bold green]",
                subtitle=f"Experts: {list(response.expert_contributions.keys())}"
            ))

            # Show stats
            console.print(f"\n[dim]Tokens: {response.total_tokens} | Latency: {response.total_latency_ms:.0f}ms[/dim]")

            await swarm_instance.shutdown()

        asyncio.run(run_query())

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@swarm.command("train")
@click.option("--expert", "-e", required=True, help="Expert name to train")
@click.option("--domain-config", "-d", required=True, help="Domain configuration file")
@click.option("--output", "-o", default="models/experts", help="Output directory")
@click.option("--epochs", default=10, help="Number of training epochs")
@click.option("--preset", "-p", default="500m", help="Model preset to use")
def swarm_train(expert: str, domain_config: str, output: str, epochs: int, preset: str):
    """
    Train an expert model on a specific domain.

    Example:
        jupiter swarm train -e python-core -d config/domains/python.yaml
    """
    from jupiter.swarm import ExpertConfig
    from jupiter.config.domain import DomainConfig
    from jupiter.training.model.config import ModelConfig
    from jupiter.training.model.architecture import JupiterModel
    import yaml

    console.print(Panel.fit(
        f"[bold blue]Training Expert:[/bold blue] {expert}",
        subtitle=f"Domain: {domain_config}"
    ))

    try:
        # Load domain config
        with open(domain_config) as f:
            domain = DomainConfig.from_dict(yaml.safe_load(f))

        # Create model
        model_config = ModelConfig.from_preset(preset)
        console.print(f"Model: {model_config.name} ({model_config.num_parameters_str})")

        output_path = Path(output) / expert
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"Output: {output_path}")
        console.print(f"Epochs: {epochs}")

        # TODO: Implement actual training loop
        console.print("\n[yellow]Note: Full training pipeline coming soon![/yellow]")
        console.print("For now, use 'jupiter start' to train the base model,")
        console.print("then configure it as an expert.")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@swarm.command("create-expert")
@click.argument("name")
@click.option("--domain", "-d", required=True, help="Expert domain (e.g., python, react)")
@click.option("--output", "-o", default="config/experts", help="Output directory")
def swarm_create_expert(name: str, domain: str, output: str):
    """
    Create a new expert configuration from template.

    Example:
        jupiter swarm create-expert my-expert --domain python
    """
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    expert_file = output_path / f"{name}.yaml"

    if expert_file.exists():
        console.print(f"[red]Error:[/red] Expert '{name}' already exists")
        return

    template = f'''# =============================================================================
# EXPERT: {name}
# =============================================================================

name: "{name}"
domain: "{domain}"
description: "Expert in {domain}"

base_model: "jupiter-500m"
model_path: null  # Set after training

keywords:
  - "{domain}"
  # Add more keywords...

capabilities:
  - "{domain.title()} fundamentals"
  # Add more capabilities...

max_tokens: 1024
temperature: 0.7
top_p: 0.9

system_prompt: |
  You are an expert in {domain}.

  Provide detailed, accurate, and practical responses.
  Always include examples when relevant.
'''

    with open(expert_file, 'w') as f:
        f.write(template)

    console.print(f"[green]✓[/green] Expert created: {expert_file}")
    console.print(f"  Edit the file to customize your expert configuration")


@swarm.command("status")
@click.option("--experts", "-e", default="config/experts", help="Experts configuration directory")
def swarm_status(experts: str):
    """
    Show swarm system status.
    """
    from jupiter.swarm import Swarm, SwarmConfig

    console.print("[bold]Jupiter Swarm Status[/bold]\n")

    # Check experts directory
    experts_path = Path(experts)
    if experts_path.exists():
        expert_files = list(experts_path.rglob("*.yaml"))
        expert_files = [f for f in expert_files if not f.name.startswith("_")]
        console.print(f"Expert configurations: [green]{len(expert_files)}[/green]")
    else:
        console.print(f"[yellow]Experts directory not found: {experts}[/yellow]")

    # Check for trained models
    models_path = Path("models/experts")
    if models_path.exists():
        trained = list(models_path.iterdir())
        console.print(f"Trained expert models: [green]{len(trained)}[/green]")
        for model_dir in trained:
            if model_dir.is_dir():
                console.print(f"  - {model_dir.name}")
    else:
        console.print("Trained expert models: [dim]0[/dim]")

    # Check MLX availability
    try:
        import mlx.core as mx
        console.print(f"\nMLX: [green]Available[/green] (v{mx.__version__})")
        console.print(f"  Device: {mx.default_device()}")
    except ImportError:
        console.print("\nMLX: [red]Not installed[/red]")


if __name__ == "__main__":
    main()
