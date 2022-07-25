# TMbed - Transmembrane proteins predicted through Language Model embeddings

TMbed predicts transmembrane beta barrel and alpha helical proteins, the position and orientation of their transmembrane segments, and signal peptides.
We use a Protein Language Model, [ProtT5-XL-U50](https://github.com/agemagician/ProtTrans) [1], to generate embeddings used as input for our method.

Pre-Print: [bioRxiv](https://doi.org/10.1101/2022.06.12.495804)\
Publication: N.A.

TMbed is also available via [bio_embeddings](https://github.com/sacdallago/bio_embeddings).\
Or you can try out TMbed using [Google Colab](https://colab.research.google.com/drive/1FbT2rQHYT67NNHCrGw4t0WMEHCY9lqR2?usp=sharing).


# Table of Contents
1. [Install](#install)
2. [Usage](#usage)
3. [Prediction output](#prediction-output)
4. [Precomputed predictions](#precomputed-predictions)
5. [Roadmap](#roadmap)
6. [References](#references)


## Install

1. Clone the repository and run directly with `python -m tmbed`.

    ```bash
    git clone https://github.com/BernhoferM/TMbed.git tmbed
    cd tmbed/
    python -m tmbed --help
    ```

2. Clone the repository and install with `pip` or `poetry`.

    ```bash
    git clone https://github.com/BernhoferM/TMbed.git tmbed
    cd tmbed/
    pip install .
    tmbed --help
    ```

3. Directly install from the repository.

    ```bash
    pip install git+https://github.com/BernhoferM/TMbed.git
    tmbed --help
    ```


### PyTorch (GPU or CPU only)

If you want to use GPU acceleration (highly recommended), please install the corresponding version of PyTorch: [Get Started](https://pytorch.org/get-started/locally/)

*Reproducibility:* TMbed tries to make predictions on GPU as deterministic as possible.\
However, some things are left to the user, such as setting the [CUBLAS_WORKSPACE_CONFIG](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility) EnvVar.\
Regarding reproducibility in PyTorch, we also recommend their [notes on randomness and reproducibility](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility).


### Requirements
```
h5py >= 3.2.1
numpy >= 1.20.3
sentencepiece >= 0.1.96
torch >= 1.10.2
tqdm >= 4.62.3
transformers >= 4.11.3
typer >= 0.4.1
```


## Usage

TMbed has two commands `embed` and `predict` that you can use to generate embeddings and predictions.


### First run

The first time TMbed is used to generate embeddings, it will automatically download the [ProtT5-XL-U50](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc) encoder model (2.25 GB) and save it inside the `models/t5/` subdirectory.


### Generate embeddings for a set of protein sequences

You can generate embeddings for a set of proteins using the `embed` command.\
The only required input is a FASTA file containing the protein sequences.

```bash
python -m tmbed embed -f sample.fasta -e sample.h5
```

The created HDF5 file is indexed with the md5 hash sums of the corresponding protein sequences.\
Every entry contains a `L x 1024` matrix, where `L` is the length of the protein sequence.


### Predict transmembrane proteins and segments

With the `predict` command you can generate predictions for a set of proteins.\
The only required input is a FASTA file containing the protein sequences. TMbed will generate the needed embeddings on the fly. If you also supply a file with embeddings, those embeddings will be used and only the subset of proteins contained within both input files will be predicted.

```bash
python -m tmbed predict -f sample.fasta -p sample.pred
```

```bash
python -m tmbed predict -f sample.fasta -e sample.h5 -p sample.pred
```


### Optional arguments

`--out-format` sets the output format for the prediction file.

`--batch-size` is an approximation of how many residues should be included per batch.\
Each batch is constrained by ***N \* L<sup>1.5</sup> &le; BS<sup>1.5</sup>***, where ***N*** is the number of sequences in the batch, ***L*** is the length of the longest sequence in the batch, and ***BS*** is the batch size. Batches with only a single sequence can break this restriction.

`--use-gpu / --no-use-gpu` controls whether TMbed will try to use an available GPU to speed up computations.

`--cpu-fallback / --no-cpu-fallback` controls whether TMbed will try to use the CPU if it fails to compute the embeddings on GPU.


### Hardware requirements

When in half-precision mode, the ProtT5-XL-U50 encoder needs about 2.5 GB of VRAM on the GPU.

Additional memory requirements to generate embeddings depend heavily on the sequence length.\
We recommend a GPU with at least 12GB of VRAM, which is enough for sequences of up to \~4200 residues.

If you run into "out of memory" issues, try reducing the batch size.


## Prediction output

TMbed supports four different output formats:
- `0`: 3-line format with directed segments.
- `1`: 3-line format with undirected segments.
- `2`: Tabular format with directed segments.
- `3`: Tabular format with undirected segments.

Predicted residue classes are encoded by single letters.\
In 3-line format, every protein is represented by three lines: header, sequence, labels.\
In tabular format, every protein is represented by a table containing sequence, labels, and class probabilities.

1. `--out-format=0 (default)`

    - `B`: Transmembrane beta strand (IN-->OUT orientation)
    - `b`: Transmembrane beta strand (OUT-->IN orientation)
    - `H`: Transmembrane alpha helix (IN-->OUT orientation)
    - `h`: Transmembrane alpha helix (OUT-->IN orientation)
    - `S`: Signal peptide
    - `.`: Non-Transmembrane

    ```
    >7acg_A|P18895|ALGE_PSEAE
    MNSSRSVNPRPSFAPRALSLAIALLLGAPAFAANSGEAPKNFGLDVKITGESENDRDLGTAPGGTLNDIGIDLRPWAFGQWGDWSAYFMGQAVAATDTIETDTLQSDTDDGNNSRNDGREPDKSYLAAREFWVDYAGLTAYPGEHLRFGRQRLREDSGQWQDTNIEALNWSFETTLLNAHAGVAQRFSEYRTDLDELAPEDKDRTHVFGDISTQWAPHHRIGVRIHHADDSGHLRRPGEEVDNLDKTYTGQLTWLGIEATGDAYNYRSSMPLNYWASATWLTGDRDNLTTTTVDDRRIATGKQSGDVNAFGVDLGLRWNIDEQWKAGVGYARGSGGGKDGEEQFQQTGLESNRSNFTGTRSRVHRFGEAFRGELSNLQAATLFGSWQLREDYDASLVYHKFWRVDDDSDIGTSGINAALQPGEKDIGQELDLVVTKYFKQGLLPASMSQYVDEPSALIRFRGGLFKPGDAYGPGTDSTMHRAFVDFIWRF
    SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS.........BBBBBBBBBB.................bbbbbbbbbbb.....BBBBBBBBBB...............................bbbbbbbbbb.........BBBBBB...............bbbbbbbb....BBBBBBBB....................bbbbbbbb......BBBBBBBB..........................bbbbbbbb..........BBBBBBBBBB............................bbbbbbbbbb.....BBBBBBBB..............................................bbbbbbbbb.....BBBBBBBB............................bbbbbbbbb..................BBBBBBBBBB...............bbbbbbbbb.
    ```

2. `--out-format=1`

    - `B`: Transmembrane beta strand
    - `H`: Transmembrane alpha helix
    - `S`: Signal peptide
    - `i`: Non-Transmembrane, inside
    - `o`: Non-Transmembrane, outside

    ```
    >7acg_A|P18895|ALGE_PSEAE
    MNSSRSVNPRPSFAPRALSLAIALLLGAPAFAANSGEAPKNFGLDVKITGESENDRDLGTAPGGTLNDIGIDLRPWAFGQWGDWSAYFMGQAVAATDTIETDTLQSDTDDGNNSRNDGREPDKSYLAAREFWVDYAGLTAYPGEHLRFGRQRLREDSGQWQDTNIEALNWSFETTLLNAHAGVAQRFSEYRTDLDELAPEDKDRTHVFGDISTQWAPHHRIGVRIHHADDSGHLRRPGEEVDNLDKTYTGQLTWLGIEATGDAYNYRSSMPLNYWASATWLTGDRDNLTTTTVDDRRIATGKQSGDVNAFGVDLGLRWNIDEQWKAGVGYARGSGGGKDGEEQFQQTGLESNRSNFTGTRSRVHRFGEAFRGELSNLQAATLFGSWQLREDYDASLVYHKFWRVDDDSDIGTSGINAALQPGEKDIGQELDLVVTKYFKQGLLPASMSQYVDEPSALIRFRGGLFKPGDAYGPGTDSTMHRAFVDFIWRF
    SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSiiiiiiiiiBBBBBBBBBBoooooooooooooooooBBBBBBBBBBBiiiiiBBBBBBBBBBoooooooooooooooooooooooooooooooBBBBBBBBBBiiiiiiiiiBBBBBBoooooooooooooooBBBBBBBBiiiiBBBBBBBBooooooooooooooooooooBBBBBBBBiiiiiiBBBBBBBBooooooooooooooooooooooooooBBBBBBBBiiiiiiiiiiBBBBBBBBBBooooooooooooooooooooooooooooBBBBBBBBBBiiiiiBBBBBBBBooooooooooooooooooooooooooooooooooooooooooooooBBBBBBBBBiiiiiBBBBBBBBooooooooooooooooooooooooooooBBBBBBBBBiiiiiiiiiiiiiiiiiiBBBBBBBBBBoooooooooooooooBBBBBBBBBi
    ```

2. `--out-format=2` and `--out-format=3`

    - `AA`: Amino acid
    - `PRD`: Predicted class label
    - `P(B)`: Probability for class 'transmembrane beta strand'
    - `P(H)`: Probability for class 'transmembrane alpha helix'
    - `P(S)`: Probability for class 'signal peptide'
    - `P(i)`: Probability for class 'non-transmembrane, inside'
    - `P(o)`: Probability for class 'non-transmembrane, outside'

    `--out-format=2` uses the same class labels as `--out-format=0`.\
    `--out-format=3` uses the same class labels as `--out-format=1`.

    ```
    >7acg_A|P18895|ALGE_PSEAE
    AA  PRD P(B)    P(H)    P(S)    P(i)    P(o)
    M   S   0.00    0.00    0.94    0.05    0.00
    N   S   0.00    0.00    0.98    0.02    0.00
    S   S   0.00    0.00    0.99    0.01    0.00
    S   S   0.00    0.00    0.99    0.01    0.00
    R   S   0.00    0.00    1.00    0.00    0.00
    S   S   0.00    0.00    1.00    0.00    0.00
    V   S   0.00    0.00    0.99    0.00    0.00
    N   S   0.00    0.00    0.99    0.01    0.00
    ...
    ```


## Precomputed predictions

We provide precomputed predictions for the human proteome and for UniProtKB/Swiss-Prot.

- Human (21-04-2022): [Download](https://rostlab.org/public/tmbed/predictions/human_210422_tmbed.tar.gz)
- UniProtKB/Swiss-Prot (11-05-2022): [Download](https://rostlab.org/public/tmbed/predictions/swissprot_110522_tmbed.tar.gz)


## Roadmap

- [x] Install via GitHub
- [ ] Publish pypi package
- [x] Add data sets to GitHub
- [x] Create Google Colab Notebook
- [ ] Add training scripts to GitHub
- [x] Integrate into [bio_embeddings](https://github.com/sacdallago/bio_embeddings)
- [ ] Integrate into [PredictProtein](https://predictprotein.org/)


## References

[1] Elnaggar A, Heinzinger M, Dallago C, Rihawi G, Wang Y, Jones L, Gibbs T, Feher T, Angerer C, Bhowmik D, Rost B (2021). ProtTrans: Towards Cracking the Language of Lifes Code Through Self-Supervised Deep Learning and High Performance Computing. IEEE Transactions on Pattern Analysis and Machine Intelligence. doi: 10.1109/TPAMI.2021.3095381.
