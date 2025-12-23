#!/usr/bin/env python3
"""
Script de training distribuido para Jupiter.

Este script se ejecuta en cada nodo del cluster via MPI.

Uso:
    mpirun -np 5 -hostfile hostfile python scripts/train_distributed.py --domain unreal_engine
"""

import argparse
import json
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Jupiter Distributed Training")
    parser.add_argument("--domain", "-d", required=True, help="Nombre del dominio")
    parser.add_argument("--config", "-c", default="config", help="Directorio de config")
    parser.add_argument("--checkpoint", help="Checkpoint para continuar")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    # Importar después de parsear args para acelerar --help
    from jupiter.config import JupiterConfig
    from jupiter.training.distributed import DistributedTrainer
    from jupiter.data.queue import DataQueue

    # Cargar configuración
    config_path = Path(args.config)
    config = JupiterConfig.from_yaml(config_path, args.domain)

    # Override con argumentos
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr

    # Crear trainer
    trainer = DistributedTrainer(config)

    # Cargar checkpoint si existe
    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))

    # Crear cola de datos
    data_queue = DataQueue(
        data_dir=config.data_dir / "queue",
        batch_size=config.training.batch_size,
    )

    # Cargar datos
    import asyncio

    asyncio.run(data_queue.load_from_directories(
        real_dir=config.data_dir / "raw",
        synthetic_dir=config.data_dir / "synthetic",
    ))

    # Training loop
    for epoch in range(args.epochs):
        if trainer.is_main_process:
            print(f"\n=== Época {epoch + 1}/{args.epochs} ===")

        metrics = trainer.train_epoch(data_queue)

        if trainer.is_main_process:
            print(f"Loss: {metrics.loss:.4f}")
            print(f"Tokens/s: {metrics.tokens_per_second:.0f}")

            # Guardar estado del nodo
            status = {
                "step": trainer.state.step,
                "epoch": epoch + 1,
                "loss": metrics.loss,
                "samples": trainer.state.total_tokens // config.training.max_seq_length,
            }
            with open("/tmp/jupiter_node_status.json", "w") as f:
                json.dump(status, f)

    # Guardar checkpoint final
    if trainer.is_main_process:
        trainer.save_checkpoint()
        print("\nTraining completado!")


if __name__ == "__main__":
    main()
