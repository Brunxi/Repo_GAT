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

## Flujo completo (de FASTA a explicaciones)

1. **Instalar el paquete (una vez)**

   ```bash
   cd ~/Repo_GAT
   python -m pip install --user -e .
   ```

2. **Preparar los datos** (gene list, embeddings, splits y grafos). Ejemplo con el pipeline base (ESM-2):

   ```bash
   PATHO_FASTA=/ruta/PAT.fasta \
   NON_PATHO_FASTA=/ruta/NOPAT.fasta \
   sbatch scripts/prepare_data.sh
   ```

   > Para la variante de ablación con ESM-1b usa `scripts/prepare_data_fungi2.sh` y `configs/fungi2.yaml`.

3. **Entrenar** con tus hiperparámetros (ajusta `configs/fungi.yaml` o exporta variables antes del `sbatch`):

   ```bash
   WANDB_API_KEY="tu_token" sbatch scripts/run_training.sh
   ```

4. **Inferencia para una secuencia** (usa el fold cuyo checkpoint quieras):

   ```bash
   SEQ=$(awk -F'\t' '$2=="pectate_liase_bcin"{print $3}' data/fungi/orig_sample_list/gene_list.txt)

   sbatch scripts/infer_single.sh \
     "$SEQ" \
     pectate_liase_bcin \
     0 \
     configs/fungi.yaml
   ```

5. **Explicabilidad con GNNExplainer** (mismo fold y secuencia):

   ```bash
   sbatch scripts/chat_exp.sh \
     "$SEQ" \
     pectate_liase_bcin \
     0 \
     configs/fungi.yaml
   ```

6. **(Opcional) Búsqueda de hiperparámetros**

   Grid sweep:

   ```bash
   WANDB_API_KEY=... sbatch scripts/run_hparam_sweep.sh
   ```

   Optuna (búsqueda bayesiana):

   ```bash
   python -m pip install --user -e .[hpo]
   WANDB_API_KEY=... sbatch scripts/run_hparam_optuna.sh
   ```

Cada entrenamiento genera la metadata del checkpoint (`*.meta.json`), de modo que inferencia y explicaciones usan siempre los parámetros reales (dropout, ratio, etc.).
