<a target="_blank" href="https://colab.research.google.com/drive/1FbT2rQHYT67NNHCrGw4t0WMEHCY9lqR2?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# TMbed - Transmembrane proteins predicted through Language Model embeddings

TMbed predicts transmembrane beta barrel and alpha helical proteins, the position and orientation of their transmembrane segments, and signal peptides.
We use a Protein Language Model, [ProtT5-XL-U50](https://github.com/agemagician/ProtTrans) [1], to generate embeddings used as input for our method.

Pre-Print: [bioRxiv](https://doi.org/10.1101/2022.06.12.495804)\
Publication: [BMC Bioinformatics](https://doi.org/10.1186/s12859-022-04873-x)

TMbed is also available via [bio_embeddings](https://github.com/sacdallago/bio_embeddings) and [LambdaPP](https://embed.predictprotein.org/) [2].\
Or you can try out TMbed using [Google Colab](https://colab.research.google.com/drive/1FbT2rQHYT67NNHCrGw4t0WMEHCY9lqR2?usp=sharing).

Visit [TMVisDB](https://tmvisdb.predictprotein.org) [3] to see precomputed predictions for AlphaFold DB [4] structures.

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

_Reproducibility:_ TMbed tries to make predictions on GPU as deterministic as possible.\
However, some things are left to the user, such as setting the [CUBLAS_WORKSPACE_CONFIG](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) EnvVar.\
Regarding reproducibility in PyTorch, we also recommend their [notes on randomness and reproducibility](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility).

### Requirements

```txt
python >= "3.9"
h5py >= "3.2.1"
numpy >= "1.20.3"
sentencepiece >= "0.1.96"
torch >= "1.10.2"
tqdm >= "4.62.3"
transformers >= "4.11.3"
typer >= "0.4.1"
```

## Usage

TMbed has two commands `embed` and `predict` that you can use to generate embeddings and predictions.

### First run

The first time TMbed is used to generate embeddings, it will automatically download the [ProtT5-XL-U50](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc) encoder model (2.25 GB) and save it inside the `models/t5/` subdirectory.

Alternatively, you can use the `download` command to download the ProtT5 model without generating embeddings.

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
Each batch is constrained by **_N \* L<sup>1.5</sup> &le; BS<sup>1.5</sup>_**, where **_N_** is the number of sequences in the batch, **_L_** is the length of the longest sequence in the batch, and **_BS_** is the batch size. Batches with only a single sequence can break this restriction.

`--use-gpu / --no-use-gpu` controls whether TMbed will try to use an available GPU to speed up computations.

`--cpu-fallback / --no-cpu-fallback` controls whether TMbed will try to use the CPU if it fails to compute the embeddings on GPU.

### Hardware requirements

When in half-precision mode, the ProtT5-XL-U50 encoder needs about 2.5 GB of VRAM on the GPU.

Additional memory requirements to generate embeddings depend heavily on the sequence length.\
We recommend a GPU with at least 12GB of VRAM, which is enough for sequences of up to \~4200 residues.

If you run into "out of memory" issues, try reducing the batch size.

## Prediction output

TMbed supports five different output formats:

-   `0`: 3-line format with directed segments.
-   `1`: 3-line format with undirected segments.
-   `2`: Tabular format with directed segments.
-   `3`: Tabular format with undirected segments.
-   `4`: 3-line format with directed segments and explicit inside/outside prediction (a mix of format `0` and `1`).

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

3. `--out-format=2` and `--out-format=3`

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

4. `--out-format=4`

    - `B`: Transmembrane beta strand (IN-->OUT orientation)
    - `b`: Transmembrane beta strand (OUT-->IN orientation)
    - `H`: Transmembrane alpha helix (IN-->OUT orientation)
    - `h`: Transmembrane alpha helix (OUT-->IN orientation)
    - `S`: Signal peptide
    - `i`: Non-Transmembrane, inside
    - `o`: Non-Transmembrane, outside

    ```
    >7acg_A|P18895|ALGE_PSEAE
    MNSSRSVNPRPSFAPRALSLAIALLLGAPAFAANSGEAPKNFGLDVKITGESENDRDLGTAPGGTLNDIGIDLRPWAFGQWGDWSAYFMGQAVAATDTIETDTLQSDTDDGNNSRNDGREPDKSYLAAREFWVDYAGLTAYPGEHLRFGRQRLREDSGQWQDTNIEALNWSFETTLLNAHAGVAQRFSEYRTDLDELAPEDKDRTHVFGDISTQWAPHHRIGVRIHHADDSGHLRRPGEEVDNLDKTYTGQLTWLGIEATGDAYNYRSSMPLNYWASATWLTGDRDNLTTTTVDDRRIATGKQSGDVNAFGVDLGLRWNIDEQWKAGVGYARGSGGGKDGEEQFQQTGLESNRSNFTGTRSRVHRFGEAFRGELSNLQAATLFGSWQLREDYDASLVYHKFWRVDDDSDIGTSGINAALQPGEKDIGQELDLVVTKYFKQGLLPASMSQYVDEPSALIRFRGGLFKPGDAYGPGTDSTMHRAFVDFIWRF
    SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSiiiiiiiiiBBBBBBBBBBooooooooooooooooobbbbbbbbbbbiiiiiBBBBBBBBBBooooooooooooooooooooooooooooooobbbbbbbbbbiiiiiiiiiBBBBBBooooooooooooooobbbbbbbbiiiiBBBBBBBBoooooooooooooooooooobbbbbbbbiiiiiiBBBBBBBBoooooooooooooooooooooooooobbbbbbbbiiiiiiiiiiBBBBBBBBBBoooooooooooooooooooooooooooobbbbbbbbbbiiiiiBBBBBBBBoooooooooooooooooooooooooooooooooooooooooooooobbbbbbbbbiiiiiBBBBBBBBoooooooooooooooooooooooooooobbbbbbbbbiiiiiiiiiiiiiiiiiiBBBBBBBBBBooooooooooooooobbbbbbbbbi
    ```

## Precomputed predictions

We provide precomputed predictions for the human proteome and for UniProtKB/Swiss-Prot.

-   Human (21-04-2022): [Zenodo](https://zenodo.org/records/14705941)
-   UniProtKB/Swiss-Prot (11-05-2022): [Zenodo](https://zenodo.org/records/14705941)

## Roadmap

-   [x] Install via GitHub
-   [ ] Publish pypi package
-   [x] Add data sets to GitHub
-   [x] Create Google Colab Notebook
-   [ ] Add training scripts to GitHub
-   [x] Integrate into [bio_embeddings](https://github.com/sacdallago/bio_embeddings)
-   [x] Integrate into [LambdaPP](https://embed.predictprotein.org/)

## References

[1] Elnaggar A, Heinzinger M, Dallago C, Rihawi G, Wang Y, Jones L, Gibbs T, Feher T, Angerer C, Bhowmik D, Rost B (2021). ProtTrans: Towards Cracking the Language of Lifes Code Through Self-Supervised Deep Learning and High Performance Computing. IEEE Transactions on Pattern Analysis and Machine Intelligence. doi: 10.1109/TPAMI.2021.3095381.

[2] Olenyi T, Marquet C, Heinzinger M, Kröger B, Nikolova T, Bernhofer M, Sändig P, Schütze K, Littmann M, Mirdita M, Steinegger M, Dallago C, Rost B (2023). LambdaPP: Fast and accessible protein-specific phenotype predictions. Protein Sci, 32, 1:e4524.

[3] Olenyi T, Marquet C, Grekova A, Houri L, Heinzinger M, Dallago C, Rost B (2024). TMVisDB: Annotation and 3D-visualization of transmembrane proteins. bioRxiv, 2024.11.22.624323.

[4] Varadi M, Anyango S, Deshpande M, Nair S, Natassia C, Yordanova G, Yuan D, Stroe O, Wood G, Laydon A, Zidek A, Green T, Tunyasuvunakool K, Petersen S, Jumper J, Clancy E, Green R, Vora A, Lutfi M, Figurnov M, Cowie A, Hobbs N, Kohli P, Kleywegt G, Birney E, Hassabis D, Velankar S (2022). AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models. Nucleic Acids Res, 50, D1:D439-D444.
