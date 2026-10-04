#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Override with: PYTHON=/path/to/python sbatch py_localknn.sh <input_dir> <output_dir> [k_list] [n_points] [train_frac] [early_stop] [pattern]
PYTHON=${PYTHON:-/mnt/beegfs/home/juan.vidal/miniforge3/envs/covroc/bin/python3.10}

# El coste es casi todo el ajuste de las 4 redes sobre la mitad de los datos (~35-40s a
# n=5000); la parte local (ordenar vecinos + DeLong) es despreciable. Cuenta ~1h por
# escenario para 100 replicas, frente a ~3.3h del sandwich puntual.
#
# $1: carpeta del escenario (e.g. input_real_2/scenario_1)
# $2: carpeta de salida
# $3: k_list, separada por comas (default "5,10,15,20,50,100,300,1000"). Barrer un ORDEN DE
#     MAGNITUD es el diagnostico de sesgo: valores cercanos solo miden varianza.
# $4: n_points, sujetos objetivo por replica (default 100)
# $5: train_frac (default 0.5)
# $6: early_stop_patience (default 40)
# $7: pattern (default solo n=5000)
# $8: ci_levels, separados por comas (default "0.95,0.8,0.5")
#     Para una banda SIMULTANEA sobre los G puntos de la rejilla por Sidak, pide ademas el
#     nivel puntual alpha^(1/G). Salen todos del mismo ajuste, asi que anadir niveles no
#     cuesta tiempo. Sidak es casi exacto aqui, no conservador, porque los puntos de la
#     rejilla resultaron practicamente independientes con G=11 (la simultanea sin corregir,
#     0.443, coincide con el 0.434 que predice la independencia). Con G mayor los puntos se
#     juntan y pueden correlacionarse; entonces Sidak se vuelve conservador, no invalido.
# $9: grid_points, puntos de la rejilla fija (default 11)
#     OJO: $8 y $9 van atados. Niveles por G:
#       G=11 -> 0.99535,0.97992,0.93893
#       G=21 -> 0.99756,0.98943,0.96753
#       G=31 -> 0.99835,0.99283,0.97789
#     para simultaneas de 0.95 / 0.80 / 0.50 respectivamente.
# $10: localize -- model | covariate (default model)
#     Donde se toma el vecindario. "model" en (mu_hat, sigma_hat) es lo validado
#     hasta ahora y lo unico viable con 4 covariables; "covariate" lo toma en x
#     directamente. El caso NHANES necesita el segundo, asi que hay que medir
#     los dos sobre los mismos escenarios para poder usar uno solo en el articulo.
INPUT_DIR=$1
OUTPUT_DIR=$2
K_LIST=${3:-5,10,15,20,50,100,300,1000}
N_POINTS=${4:-100}
TRAIN_FRAC=${5:-0.5}
EARLY_STOP=${6:-40}
PATTERN=${7:-_5000_.*_data\.csv$}
CI_LEVELS=${8:-0.95,0.8,0.5}
GRID_POINTS=${9:-11}
LOCALIZE=${10:-model}

mkdir -p "$OUTPUT_DIR"
command="$PYTHON src/simulation/coverage_local_knn.py -i $INPUT_DIR -o $OUTPUT_DIR -e 800 -es $EARLY_STOP -kl $K_LIST -np $N_POINTS -tf $TRAIN_FRAC -ci $CI_LEVELS -gp $GRID_POINTS -loc $LOCALIZE -pt "$PATTERN""
echo $command
$command
