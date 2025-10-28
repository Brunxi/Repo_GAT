# Repo_GAT

Repositorio reorganizado del pipeline GAT para la predicción de genes patogénicos en hongos. La estructura sigue buenas prácticas de proyectos Python: código bajo `src/`, scripts independientes y configuración declarativa.

## Estructura

```
Repo_GAT/
├── configs/           # Archivos YAML con configuraciones por especie
├── scripts/           # Scripts HPC (Slurm) que invocan la CLI
├── src/gat_pipeline/  # Paquete Python con toda la lógica
└── pyproject.toml     # Dependencias y punto de entrada gat-pipeline
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Comandos principales

```bash
# Preparar datos (FASTA → raw embeddings, particiones, grafos)
gat-pipeline prepare-data --config configs/fungi.yaml \
  --pathogenesis-fasta path/to/PAT.fasta \
  --non-pathogenesis-fasta path/to/NOPAT.fasta

# Entrenamiento por fold
gat-pipeline train-fold --config configs/fungi.yaml --fold 0 --model gat

# Inferencia de una secuencia
gat-pipeline infer-sequence --config configs/fungi.yaml \
  --sequence "MTEITAAMVK..." --name protein_X \
  --model-checkpoint experiments/fungi/gat/fold_0/best_aupr.pt

# Inferencia masiva a partir de un FASTA
gat-pipeline infer-fasta --config configs/fungi.yaml \
  --fasta proteome.fasta --output proteome_predictions.csv

# Explicaciones con GNNExplainer
gat-pipeline explain-nodes --sequence "MTEITAAMVK..." --name protein_X \
  --model-checkpoint experiments/fungi/gat/fold_0/best_aupr.pt
```

Los scripts en `scripts/` son plantillas SLURM que utilizan estos comandos.

## Notas

- El módulo `gat_pipeline` centraliza la lectura de configuración, creación de embeddings ESM-2, generación de grafos, entrenamiento y tareas de inferencia.
- Los modelos GAT/GCN/SAGE y los ataques adversarios FGM están en `gat_pipeline.models`.
- Las representaciones ESM-2 se obtienen con `fair-esm` (contact head oficial), garantizando mapas de contacto reales.
- Se usa `wandb` de forma opcional; definir `WANDB_API_KEY` antes de ejecutar para habilitarlo.
