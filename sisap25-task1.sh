echo "RUN AS: bash -x task1.sh 2>&1 | tee log-task1.txt"

#PATH_TO_HOST_DIR=/home/sisap23evaluation/data2025/without-gold
PATH_TO_HOST_DIR=/home/sisap23evaluation/data2025
## WARNING RUN WITHOUT NETWORK
PATH_TO_CONTAINER_DIR=/app/data
OUT_PATH_TO_HOST_DIR=$(pwd)/results-task1
OUT_PATH_TO_CONTAINER_DIR=/app/results

mkdir $OUT_PATH_TO_HOST_DIR
echo "==== pwd: $(pwd)"
echo "==== directory listing: "
ls
echo "==== environment"
set
echo "==== RUN BEGINS $(date)"
#docker run --rm --memory=16g --memory-swap=16g -v <path to benchmark-dev-pubmed23.h5>:/app/data/benchmark-dev-pubmed23.h5:ro   -v <path to directory for result files>:/app/result doublefiltering /bin/bash /app/autoexec_all.sh
docker run \
    -it \
    --cpus=8 \
    --memory=16g \
    --network none \
    --volume $PATH_TO_HOST_DIR:$PATH_TO_CONTAINER_DIR:ro \
    --volume $OUT_PATH_TO_HOST_DIR:$OUT_PATH_TO_CONTAINER_DIR:rw \
      sisap25/hforest python sisap2025.py task1
    #--memory-swap=16g \


echo "==== RUN ENDS $(date)"


