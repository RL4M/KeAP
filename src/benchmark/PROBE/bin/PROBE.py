import yaml
import pandas as pd
import tqdm
import semantic_similarity_infer as ssi
import target_family_classifier as tfc
import function_predictor as fp
import binding_affinity_estimator as bae

print("\n\nPROBE (Protein RepresentatiOn Benchmark) run is started...\n\n")

with open('probe_config.yaml') as f:
	args = yaml.load(f, Loader=yaml.FullLoader)

if args["benchmark"] not in ["similarity","family","function","affinity","all"]:
        parser.error('At least one benchmark type should be selected')

print(args)

def load_representation(multi_col_representation_vector_file_path):
    multi_col_representation_vector = pd.read_csv(multi_col_representation_vector_file_path)
    vals = multi_col_representation_vector.iloc[:,1:(len(multi_col_representation_vector.columns))]
    original_values_as_df = pd.DataFrame({'Entry': pd.Series([], dtype='str'),'Vector': pd.Series([], dtype='object')})
    for index, row in tqdm.tqdm(vals.iterrows(), total = len(vals)):
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [multi_col_representation_vector.iloc[index]['Entry']] + [list_of_floats]
    return original_values_as_df

if args["benchmark"] in  ["similarity","function","all"]:
    print("\nRepresentation vectors are loading...\n")
    representation_dataframe = load_representation(args["representation_file_human"])
 
if args["benchmark"] in  ["similarity","all"]:
    print("\nSemantic similarity Inference Benchmark is running...\n")
    ssi.representation_dataframe = representation_dataframe
    ssi.representation_name = args["representation_name"]
    ssi.protein_names = ssi.representation_dataframe['Entry'].tolist()
    ssi.similarity_tasks = args["similarity_tasks"]
    ssi.detailed_output = args["detailed_output"]
    ssi.calculate_all_correlations()
if args["benchmark"] in  ["function","all"]:
    print("\n\nOntology-based protein function prediction benchmark is running...\n")
    fp.aspect_type = args["function_prediction_aspect"]
    fp.dataset_type = args["function_prediction_dataset"]
    fp.representation_dataframe = representation_dataframe
    fp.representation_name = args["representation_name"]
    fp.detailed_output = args["detailed_output"]
    fp.pred_output()
if args["benchmark"] in  ["family","all"]:
    print("\n\nDrug target protein family classification benchmark is running...\n")
    tfc.representation_path = args["representation_file_human"]
    tfc.representation_name = args["representation_name"]
    tfc.detailed_output = args["detailed_output"]
    for dataset in args["family_prediction_dataset"]:
	    tfc.score_protein_rep(dataset)
if args["benchmark"] in  ["affinity","all"]:
    print("\n\nProtein-protein binding affinity estimation benchmark is running...\n")
    bae.skempi_vectors_path = args["representation_file_affinity"]
    bae.representation_name = args["representation_name"]
    bae.predict_affinities_and_report_results()
print("\n\nPROBE (Protein RepresentatiOn Benchmark) run is finished...\n")


