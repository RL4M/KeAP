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

- `go_def.txt`: GO term definition described in text. The GO term name and corresponding definition were concatenated by colon.
- `go_type.txt`: The ontology type which the specific GO term belong to. The index is correponding to GO ID in `go2id.txt` file.
- `go2id.txt`: The ID mapping of GO terms.
- `go_go_triplet.txt`: GO-GO triplet data (Not used). The triplet data constitutes the interior structure of Gene Ontology. The format for each triplet is < `h r t`>, where `h` and `t` are respectively head entity and tail entity, both GO term nodes. `r` is relation between two GO terms, e.g. `is_a` and `part_of`.
- `protein_seq.txt`: Protein sequence data. The whole protein sequence data are used as inputs.
- `protein2id.txt`: The ID mapping of proteins.
- `protein_go_train_triplet.txt`: Protein-GO triplet data. The triplet data constitutes the exterior structure of Gene Ontology, i.e. Gene annotation. The format for each triplet is <`h r t`>, where `h` and `t` are respectively head entity and tail entity. Each Protein-GO triplet represent a specific gene annotation, where the head entity is a specific protein and tail entity is the corresponding annotated GO term (Attribute), e.g. protein binding function. `r` is relation between the protein and GO term.
- `protein_go_train_triplet_v2.txt`: Protein-GO triplet data filtered to remove triplets that contain protein with 100% sequence similarity with proteins in downstream tasks. We used [blastp](https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi) to identify similar proteins.
- `relation2id.txt`:  The ID mapping of relations.

For generating your own pre-training data, you need to download the following raw data:

- `go.obo`: the structure data of Gene Ontology. The download link and detailed format can be found in [Gene Ontology](http://geneontology.org/docs/download-ontology/)`
- `uniprot_sprot.dat`: protein Swiss-Prot database. [[link]](https://www.uniprot.org/downloads)
- `goa_uniprot_all.gpa`: Gene Annotation data. [[link]](https://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/)

After downloading these raw data and **configuring associated paths in `gen_onto_protein_data.py`**, you can execute the following script to generate pre-training data:

```bash
python gen_onto_protein_data.py
```
