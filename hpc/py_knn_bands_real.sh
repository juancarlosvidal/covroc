#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Override with: PYTHON=/path/to/python sbatch py_knn_bands_real.sh [k_list] [out_dir] [extra args...]
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# Bandas k-NN locales para la AUC condicional en la cohorte NHANES, en las seis
# combinaciones horizonte x sexo (--all).
#
# El coste es el ajuste de las dos redes por combinacion (3000 epocas, patience 200,
# ~2.600 filas): del orden de 1-3 min por combinacion en GPU, o sea <20 min en total.
# El bucle sobre los valores de k es gratis -- se ajusta UNA vez y despues se recorren
# todos los k, asi que pedir cinco valores no cuesta mas que pedir uno. Por eso conviene
# pedirlos todos de golpe en vez de hacer una pasada por cada k: ademas de ser mas
# rapido, garantiza que todos comparten el mismo ajuste y son comparables entre si.
#
# La semilla (--seed 42, por defecto en el script) fija el ajuste, de modo que esta
# corrida reproduce la estimacion plug-in de la red de corridas anteriores con los
# mismos hiperparametros.
#
# $1: lista de k separada por comas (default "5,10,20,30,50")
# $2: carpeta de salida          (default output/knn_bands_real_allk)
# $3+: argumentos extra que se pasan tal cual (p.ej. -loc model, -est residualized)
K_LIST=${1:-5,10,20,30,50}
OUT_DIR=${2:-output/knn_bands_real_allk}
shift 2 2>/dev/null || shift $#

# --all recorre las seis combinaciones. Si quien llama pide una sola con -g/-s no lo
# anadimos, de modo que la misma receta sirve para la prueba de humo y para la completa.
ALL="--all"
case " $* " in *" -g "*|*" -s "*) ALL="" ;; esac

mkdir -p "$OUT_DIR"
command="$PYTHON src/real_data/local_knn_bands_real.py $ALL -k $K_LIST -o $OUT_DIR $@"
echo "$command"
$command
