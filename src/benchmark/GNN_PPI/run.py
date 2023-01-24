import os

def run_func(description, ppi_path, pseq_path, vec_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs):
    os.system("python -u gnn_train.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --split_new={} \
            --split_mode={} \
            --train_valid_index_path={} \
            --use_lr_scheduler={} \
            --save_path={} \
            --graph_only_train={} \
            --batch_size={} \
            --epochs={} ".format(description, ppi_path, pseq_path, vec_path, 
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path, graph_only_train, 
                    batch_size, epochs))

if __name__ == "__main__":
    description = "KeAP20_148k_bfs_10"

    ppi_path = "data/downstream_datasets/ppi_data/protein.actions.SHS148k.STRING.txt"
    pseq_path = "data/downstream_datasets/ppi_data/protein.SHS148k.sequences.dictionary.tsv"
    vec_path = "data/downstream_datasets/ppi_data/features/protein_embedding_KeAP20_shs148k.npy"

    split_new = "True"
    split_mode = "bfs"
    train_valid_index_path = "data/downstream_datasets/ppi_data/new_train_valid_index_json/SHS148k.bfs.fold1.json"

    use_lr_scheduler = "True"
    save_path = "output/ppi"
    graph_only_train = "False"

    batch_size = 2048
    epochs = 300

    run_func(description, ppi_path, pseq_path, vec_path, 
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs)