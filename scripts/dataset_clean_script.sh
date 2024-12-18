tot=32

for I in $(seq 2 9); do
    json_file=./playground/image_caption/GBC-10M/train_${I}_clean.json
    for IDX in $(seq 0 $((tot-1))); do
        python scripts/dataset_clean_valid.py $json_file $IDX $tot &
    done

    wait

    python scripts/dataset_clean_combine.py $json_file $tot
done