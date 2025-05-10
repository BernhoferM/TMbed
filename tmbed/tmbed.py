# Copyright 2022 Rostlab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import h5py
import torch
import typer

from tqdm import tqdm
from pathlib import Path

from .model import Predictor
from .embed import T5Encoder
from .viterbi import Decoder

from .utils import ARGS, OutFmt
from .utils import seed_all, read_fasta, make_batches, collate_batch, make_mask


app = typer.Typer(help='Transmembrane protein predictor using embeddings.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init(use_gpu):
    seed_all(101)

    if not use_gpu:
        global device
        device = torch.device('cpu')

    valid_env = os.getenv('CUBLAS_WORKSPACE_CONFIG') in {':16:8', ':4096:8'}

    if not use_gpu or valid_env:
        torch.use_deterministic_algorithms(True)
    else:
        typer.echo(('Warning: EnvVar "CUBLAS_WORKSPACE_CONFIG" is neither set '
                    'to ":16:8" nor to ":4096:8". This can cause computations '
                    'on the GPU to be non-deterministic. For more information '
                    'see: https://docs.nvidia.com/cuda/cublas/index.html'
                    '#results-reproducibility\n'), err=True)


def load_encoder(use_gpu):
    if os.getenv('HF_HOME', None) is not None:
        return T5Encoder(None, use_gpu)

    root_path = Path(__file__).parent
    model_path = Path(root_path, 'models/t5/')

    return T5Encoder(model_path, use_gpu)


def load_models():
    models = []
    root_path = Path(__file__).parent
    model_path = Path(root_path, 'models/cnn/')

    for model_file in sorted(model_path.glob('*.pt')):
        model = Predictor()

        model.load_state_dict(torch.load(model_file)['model'])

        model = model.eval().requires_grad_(False).to(device)

        models.append(model)

    return models


def filter_proteins(proteins, embeddings_file):
    filtered = []

    with h5py.File(embeddings_file, mode='r') as ef:
        for protein in proteins:
            if not protein.seq_hash in ef:
                continue

            filtered.append(protein)

    return filtered


def encode_sequences(encoder, sequences, cpu_fallback):
    try:
        embeddings = encoder.embed(sequences)
    except RuntimeError as e:
        if encoder.device() == torch.device('cpu'):
            typer.echo('\n\nEmbedding on CPU failed.\n', err=True)
            raise e

        typer.echo('\n\nEmbedding on GPU failed.\n', err=True)

        if not cpu_fallback:
            raise e

        typer.echo(e, err=True)

        typer.echo('\nMoving T5Encoder to CPU.\n', err=True)

        encoder.to_cpu()

        torch.cuda.empty_cache()

        embeddings = encoder.embed(sequences)

    return embeddings


def predict_sequences(models, embeddings, mask):
    B, N, _ = embeddings.shape

    num_models = len(models)

    pred = torch.zeros((B, 5, N), device=embeddings.device)

    for model in models:
        y = model(embeddings, mask)
        pred = pred + torch.softmax(y, dim=1)

    pred = pred / num_models

    return pred


@app.command()
def download(use_gpu: bool = ARGS.use_gpu):
    '''
    Download models if necessary.
    '''
    encoder = load_encoder((use_gpu and torch.cuda.is_available()))


@app.command()
def embed(fasta_file: Path = ARGS.fasta,
          embeddings_file: Path = ARGS.emb_out,
          batch_size: int = ARGS.batch_size,
          use_gpu: bool = ARGS.use_gpu,
          cpu_fallback: bool = ARGS.cpu_fallback):
    '''
    Generate ProtT5 embeddings for a set of protein sequences.
    '''
    use_gpu = (use_gpu and torch.cuda.is_available())

    init(use_gpu)

    encoder = load_encoder(use_gpu)
    proteins = read_fasta(fasta_file)

    proteins.sort(key=lambda protein: protein.length)

    batches = make_batches(proteins, batch_size)

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(embeddings_file, mode='w') as h5f:
        with tqdm(total=len(proteins), leave=True) as progress:
            hash_set = set()

            for a, b in batches:
                batch = proteins[a:b]

                sequences = [protein.sequence for protein in batch]

                embeddings = encode_sequences(encoder, sequences, cpu_fallback)

                embeddings = embeddings.half().cpu()

                for idx, protein in enumerate(batch):
                    length = protein.length
                    seq_hash = protein.seq_hash

                    if seq_hash in hash_set:
                        continue

                    seq_emb = embeddings[idx, :length, :]

                    assert seq_emb.shape == (length, 1024)

                    h5f.create_dataset(seq_hash, data=seq_emb)

                    hash_set.add(seq_hash)

                progress.update(b - a)


@app.command()
def predict(fasta_file: Path = ARGS.fasta,
            embeddings_file: Path = ARGS.emb_in,
            output_file: Path = ARGS.predictions,
            out_format: OutFmt = ARGS.out_format,
            batch_size: int = ARGS.batch_size,
            use_gpu: bool = ARGS.use_gpu,
            cpu_fallback: bool = ARGS.cpu_fallback):
    '''
    Predict transmembrane proteins and segments using embeddings.
    If no embeddings file is supplied, embeddings are generated on the fly.
    '''
    use_gpu = (use_gpu and torch.cuda.is_available())

    init(use_gpu)

    out_format = out_format.value
    with_probabilities = out_format in {OutFmt.F2, OutFmt.F3}

    config = {'batch_size': batch_size,
              'cpu_fallback': cpu_fallback,
              'with_probabilities': with_probabilities}

    if not embeddings_file:
        encoder = load_encoder(use_gpu)

    models = load_models()
    proteins = read_fasta(fasta_file)

    if embeddings_file:
        predictions, error = predict_from_file(models,
                                               proteins,
                                               embeddings_file,
                                               config)
    else:
        predictions, error = predict_from_sequence(models,
                                                   proteins,
                                                   encoder,
                                                   config)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if out_format in {OutFmt.F0, OutFmt.F2}:
        pred_map = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'S', 5: '.', 6: '.'}
    elif out_format in {OutFmt.F1, OutFmt.F3}:
        pred_map = {0: 'B', 1: 'B', 2: 'H', 3: 'H', 4: 'S', 5: 'i', 6: 'o'}
    elif out_format in {OutFmt.F4}:
        pred_map = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'S', 5: 'i', 6: 'o'}

    if out_format in {OutFmt.F0, OutFmt.F1, OutFmt.F4}:
        write_3_line(output_file, proteins, predictions, pred_map)
    elif out_format in {OutFmt.F2, OutFmt.F3}:
        write_tabular(output_file, proteins, predictions, pred_map)

    if error:
        raise error


def predict_from_file(models, proteins, embeddings_file, config):
    decoder = Decoder()
    predictions = dict()

    proteins = filter_proteins(proteins, embeddings_file)
    proteins = sorted(proteins, key=lambda protein: protein.length)

    batches = make_batches(proteins, config['batch_size'])

    with_probabilities = config['with_probabilities']

    with tqdm(total=len(proteins), leave=True) as progress:
        for a, b in batches:
            batch = proteins[a:b]

            _, lengths, embeddings = collate_batch(batch, embeddings_file)

            embeddings = embeddings.to(device=device)
            embeddings = embeddings.to(dtype=torch.float32)

            mask = make_mask(embeddings, lengths)

            probabilities = predict_sequences(models, embeddings, mask)

            mask = mask.cpu()
            probabilities = probabilities.cpu()

            prediction = decoder(probabilities, mask).byte()

            if with_probabilities:
                probabilities = probabilities.permute(0, 2, 1)

                for idx, protein in enumerate(batch):
                    length = protein.length
                    seq_hash = protein.seq_hash
                    predictions[seq_hash] = (prediction[idx, :length],
                                             probabilities[idx, :length])
            else:
                for idx, protein in enumerate(batch):
                    length = protein.length
                    seq_hash = protein.seq_hash
                    predictions[seq_hash] = (prediction[idx, :length], None)

            progress.update(b - a)

    return predictions, None


def predict_from_sequence(models, proteins, encoder, config):
    decoder = Decoder()
    predictions = dict()

    proteins = sorted(proteins, key=lambda protein: protein.length)

    batches = make_batches(proteins, config['batch_size'])

    cpu_fallback = config['cpu_fallback']
    with_probabilities = config['with_probabilities']

    with tqdm(total=len(proteins), leave=True) as progress:
        for a, b in batches:
            batch = proteins[a:b]

            lengths = [protein.length for protein in batch]
            sequences = [protein.sequence for protein in batch]

            try:
                embeddings = encode_sequences(encoder, sequences, cpu_fallback)
            except RuntimeError as e:
                typer.echo('\nAborting and printing results.\n', err=True)

                return predictions, e

            embeddings = embeddings.to(device=device)
            embeddings = embeddings.to(dtype=torch.float32)

            mask = make_mask(embeddings, lengths)

            probabilities = predict_sequences(models, embeddings, mask)

            mask = mask.cpu()
            probabilities = probabilities.cpu()

            prediction = decoder(probabilities, mask).byte()

            if with_probabilities:
                probabilities = probabilities.permute(0, 2, 1)

                for idx, protein in enumerate(batch):
                    length = protein.length
                    seq_hash = protein.seq_hash
                    predictions[seq_hash] = (prediction[idx, :length],
                                             probabilities[idx, :length])
            else:
                for idx, protein in enumerate(batch):
                    length = protein.length
                    seq_hash = protein.seq_hash
                    predictions[seq_hash] = (prediction[idx, :length], None)

            progress.update(b - a)

    return predictions, None


def write_3_line(output_file, proteins, predictions, pred_map):
    with output_file.open('w') as of:
        for protein in proteins:
            seq_hash = protein.seq_hash

            if seq_hash not in predictions:
                continue

            header = protein.header
            sequence = protein.sequence

            prediction, probabilities = predictions[seq_hash]

            prediction = ''.join(pred_map[v] for v in prediction.tolist())

            assert len(prediction) == len(sequence)

            of.write(f'{header}\n')
            of.write(f'{sequence}\n')
            of.write(f'{prediction}\n')


def write_tabular(output_file, proteins, predictions, pred_map):
    col_head = 'AA\tPRD\tP(B)\tP(H)\tP(S)\tP(i)\tP(o)'

    with output_file.open('w') as of:
        for protein in proteins:
            seq_hash = protein.seq_hash

            if seq_hash not in predictions:
                continue

            header = protein.header
            sequence = protein.sequence

            prediction, probabilities = predictions[seq_hash]

            prediction = ''.join(pred_map[v] for v in prediction.tolist())

            probabilities = probabilities.tolist()

            assert len(prediction) == len(sequence)
            assert len(probabilities) == len(prediction)

            of.write(f'{header}\n')
            of.write(f'{col_head}\n')

            for aa, prd, probs in zip(sequence, prediction, probabilities):
                probs = '\t'.join(f'{v:.2f}' for v in probs)

                of.write(f'{aa}\t{prd}\t{probs}\n')


def run():
    app()


if __name__ == '__main__':
    run()
