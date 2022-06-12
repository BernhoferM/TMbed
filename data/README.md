# TMbed - Data Sets

Data sets used during the development of TMbed.

All sequences have been collected from [OPM](https://opm.phar.umich.edu/) [1] and [UniProtKB](https://www.uniprot.org/) [2] between Jan 9 and Jan 10, 2022.

The `datasets` directory contains the three annotated data sets:
- `beta.fasta`: Transmembrane beta barrel proteins
- `alpha.fasta`: Transmembrane alpha helical proteins
- `signalp.fasta`: Globular proteins from the [SignalP 6.0](https://services.healthtech.dtu.dk/service.php?SignalP) [3] dataset
- `all_seq.fasta`: Sequences for all data sets (without per-residue annotations)

The `cv` directory contains the individual cross-validation splits, once with annotations and once sequences only.


## Annotations

Each residue is annotated with a single letter representing its class:
- `B`/`b`: Transmembrane beta strand
- `H`/`h`: Transmembrane alpha helix
- `S`: Signal peptide
- `1`: Non-Transmembrane, inside
- `2`: Non-Transmembrane, outside
- `U`: Unknown/Unresolved in PDB


## References

[1] Lomize MA, Pogozheva ID, Joo H, Mosberg HI, Lomize AL (2012). OPM database and PPM web server: resources for positioning of proteins in membranes. Nucleic Acids Res, 40, Database issue:D370-6.

[2] UniProt Consortium (2021). UniProt: the universal protein knowledgebase in 2021. Nucleic Acids Res, 49, D1:D480-D489.

[3] Teufel F, Almagro Armenteros JJ, Johansen AR, Gíslason MH, Pihl SI, Tsirigos KD, Winther O, Brunak S, von Heijne G, Nielsen H (2022). SignalP 6.0 predicts all five types of signal peptides using protein language models. Nat Biotechnol.
