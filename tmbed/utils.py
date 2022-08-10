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


import h5py
import math
import torch
import numpy
import typer
import random

from torch.nn.utils.rnn import pad_sequence

from enum import Enum
from hashlib import md5
from pathlib import Path
from dataclasses import dataclass


class OutFmt(str, Enum):
    F0 = '0'
    F1 = '1'
    F2 = '2'
    F3 = '3'
    F4 = '4'


@dataclass
class ARGS:
    fasta: Path = typer.Option(..., '--fasta', '-f',
                               help='Input FASTA file.')
    emb_in: Path = typer.Option(None, '--embedings', '-e',
                                help='Input embeddings file.')
    emb_out: Path = typer.Option(..., '--embedings', '-e',
                                 help='Output embeddings file.')
    predictions: Path = typer.Option(..., '--predictions', '-p',
                                     help='Output predictions file.')
    out_format: OutFmt = typer.Option('0', help='Prediction output format.')
    batch_size: int = typer.Option(4000, help='Approximated batch size.')
    use_gpu: bool = typer.Option(True, help='Use GPU if available.')
    cpu_fallback: bool = typer.Option(True, help='Use CPU if GPU fails.')


@dataclass
class Protein:

    header: str
    sequence: str
    length: int = 0
    seq_hash: str = ''

    def __post_init__(self):
        self.length = len(self.sequence)
        self.seq_hash = get_md5(self.sequence)


def seed_all(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_md5(string):
    md5_hash = md5()

    md5_hash.update(string.encode())

    return md5_hash.hexdigest()


def gaussian(x, std):
    pi = torch.tensor(math.pi)
    s2 = 2.0*torch.tensor(std).square()
    x2 = torch.tensor(x).square().neg()

    return torch.exp(x2 / s2) * torch.rsqrt(s2 * pi)


def gaussian_kernel(kernel_size, std=1.0):
    kernel = [gaussian(i - (kernel_size // 2), std)
              for i in range(kernel_size)]

    kernel = torch.tensor(kernel)
    kernel = kernel / kernel.sum()

    return kernel


def read_fasta(filename):
    proteins = []

    with Path(filename).open('r') as f:
        header = None
        sequence = []

        for line in f:
            line = line.strip()

            if line.startswith('>'):
                if header and sequence:
                    proteins.append(Protein(header, ''.join(sequence)))

                header = line
                sequence = []
            elif header:
                sequence.extend(line.split())

        if header and sequence:
            proteins.append(Protein(header, ''.join(sequence)))

    return proteins


def make_batches(proteins, batch_size):
    batches = []
    num_prot = 1
    last_idx = 0
    max_size = 0
    batch_size = abs(batch_size) ** 1.5

    for idx, protein in enumerate(proteins):
        seq_size = protein.length ** 1.5
        max_size = max(seq_size, max_size)

        if (idx > 0) and (num_prot * max_size > batch_size):
            batches.append((last_idx, idx))

            num_prot = 1
            last_idx = idx
            max_size = seq_size

        num_prot = num_prot + 1

    batches.append((last_idx, idx + 1))

    return batches


def collate_batch(proteins, embeddings_file):
    lengths = []
    seq_hashes = []
    embeddings = []

    with h5py.File(embeddings_file, mode='r') as h5f:
        for protein in proteins:
            lengths.append(protein.length)
            seq_hashes.append(protein.seq_hash)
            embeddings.append(torch.from_numpy(h5f[protein.seq_hash][:]))

    embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)

    return seq_hashes, lengths, embeddings


def make_mask(embeddings, lengths):
    B, N, _ = embeddings.shape

    mask = torch.zeros((B, N),
                       dtype=embeddings.dtype,
                       device=embeddings.device)

    for idx, length in enumerate(lengths):
        mask[idx, :length] = 1.0

    return mask
