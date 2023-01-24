import os


def run_func(description, ppi_path, pseq_path, vec_path,
            index_path, gnn_model, test_all):
    os.system("python gnn_test.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --index_path={} \
            --gnn_model={} \
            --test_all={} \
            ".format(description, ppi_path, pseq_path, vec_path, 
                    index_path, gnn_model, test_all))

if __name__ == "__main__":
    description = "test"

    ppi_path = "data/downstream_datasets/ppi_data/protein.actions.SHS148k.STRING.txt"
    pseq_path = "data/downstream_datasets/ppi_data/protein.SHS148k.sequences.dictionary.tsv"
    vec_path = "data/downstream_datasets/ppi_data/features/protein_embedding_KeAP20_shs148k.npy"

    index_path = "data/downstream_datasets/ppi_data/new_train_valid_index_json/SHS148k.bfs.fold1.json"
    # path to checkpoint
    dir = "output/ppi/gnn_protbert_148k_bfs_1_2022-08-22\ 13:38:11"
    gnn_model= dir + "/gnn_model_valid_best.ckpt"

    test_all = "True"

    run_func(description, ppi_path, pseq_path, vec_path, index_path, gnn_model, test_all)
