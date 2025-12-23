"""
CLI de Jupiter.

Interfaz de línea de comandos para el framework.
"""

import asyncio
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from jupiter.config import JupiterConfig
from jupiter.training.model import ModelConfig

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    Jupiter - Framework de entrenamiento distribuido para modelos de lenguaje expertos.

    Entrena modelos de 1B parámetros en clusters de Macs y/o GPUs NVIDIA.
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


if __name__ == "__main__":
    main()
