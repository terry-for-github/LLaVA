tot=32

for I in $(seq 2 9); do
    # 这里是用来过滤GBC-10M数据集时用的，要根据过滤的数据集进行修改
    json_file=./playground/image_caption/GBC-10M/train_${I}_clean.json
    for IDX in $(seq 0 $((tot-1))); do
        python scripts/dataset_clean_valid.py $json_file $IDX $tot &
    done

    wait

    python scripts/dataset_clean_combine.py $json_file $tot
done