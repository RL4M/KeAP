## Environment for ProteinKG25 Generation
**Main requirements**
- python 3.7
- biopython 1.37 
- goatools

For extracting the definition of the GO term, you need to modify the code in `goatools` library. The changes in `goatools.obo_parser` are as follows:

```python
# line 132
elif line[:5] == "def: ":
    rec_curr.definition = line[5:]

# line 169
self.definition = ""
```

## Pre-training Data Preparation

You can download [ProteinKG25](https://zjunlp.github.io/project/ProteinKG25/) from [Google Drive](https://drive.google.com/file/d/1iTC2-zbvYZCDhWM_wxRufCvV6vvPk8HR/view).

The whole compressed package includes following files:

- `go_def.txt`: GO term definition, which is text data. We concatenate GO term name and corresponding definition by colon.
- `go_type.txt`: The ontology type which the specific GO term belong to. The index is correponding to GO ID in `go2id.txt` file.
- `go2id.txt`: The ID mapping of GO terms.
- `go_go_triplet.txt`: GO-GO triplet data. The triplet data constitutes the interior structure of Gene Ontology. The data format is < `h r t`>, where `h` and `t` are respectively head entity and tail entity, both GO term nodes. `r` is relation between two GO terms, e.g. `is_a` and `part_of`.
- `protein_seq.txt`: Protein sequence data. The whole protein sequence data are used as inputs in MLM module and protein representations in KE module.
- `protein2id.txt`: The ID mapping of proteins.
- `protein_go_train_triplet.txt`: Protein-GO triplet data. The triplet data constitutes the exterior structure of Gene Ontology, i.e. Gene annotation. The data format is <`h r t`>, where `h` and `t` are respectively head entity and tail entity. It is different from GO-GO triplet that a triplet in Protein-GO triplet means a specific gene annotation, where the head entity is a specific protein and tail entity is the corresponding GO term, e.g. protein binding function. `r` is relation between the protein and GO term.
- `protein_go_train_triplet_v2.txt`: Protein-GO triplet data. Proteins that have 100% sequence similarity (identified using blastp) with protiens in downstream tasks were removed.
- `relation2id.txt`:  The ID mapping of relations. We mix relations in two triplet relation.

For generating your own pre-training data, you need to download the following raw data:

- `go.obo`: the structure data of Gene Ontology. The download link and detailed format can be found in [Gene Ontology](http://geneontology.org/docs/download-ontology/)`
- `uniprot_sprot.dat`: protein Swiss-Prot database. [[link]](https://www.uniprot.org/downloads)
- `goa_uniprot_all.gpa`: Gene Annotation data. [[link]](https://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/)

After downloading these raw data and **configuring associated paths in `gen_onto_protein_data.py`**, you can execute the following script to generate pre-training data:

```bash
python gen_onto_protein_data.py
```