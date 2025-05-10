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


import torch
import tempfile

from pathlib import Path
from transformers import logging
from transformers import T5EncoderModel, T5Tokenizer


class T5Encoder:

    def __init__(self, model_path=None, use_gpu=True):
        self.model_name = 'Rostlab/prot_t5_xl_half_uniref50-enc'

        if model_path is None:
            model_name_or_path = self.model_name
        else:
            model_name_or_path = self._download_models(model_path)

        if use_gpu and torch.cuda.is_available():
            self._load_models(model_name_or_path, torch.float16)
            self.encoder_model = self.encoder_model.cuda()
        else:
            self._load_models(model_name_or_path, torch.float32)

        self.aa_map = str.maketrans('BJOUZ', 'XXXXX')

    def _download_models(self, model_path):
        if Path(model_path, 'config.json').exists():
            return model_path

        Path(model_path).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=model_path) as temp_dir:
            model_t = T5Tokenizer.from_pretrained(self.model_name,
                                                  cache_dir=temp_dir,
                                                  do_lower_case=False)

            model_e = T5EncoderModel.from_pretrained(self.model_name,
                                                     cache_dir=temp_dir,
                                                     torch_dtype=torch.float16)

            model_t.save_pretrained(model_path)
            model_e.save_pretrained(model_path)

        del model_t, model_e

        return model_path

    def _load_models(self, model_name_or_path, dtype):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path,
                                                     do_lower_case=False)

        self.encoder_model = T5EncoderModel.from_pretrained(model_name_or_path,
                                                            torch_dtype=dtype)

        self.encoder_model = self.encoder_model.eval().requires_grad_(False)

    def device(self):
        return self.encoder_model.device

    def to_cpu(self):
        self.encoder_model = self.encoder_model.cpu().float()

    def to_cuda(self):
        self.encoder_model = self.encoder_model.half().cuda()

    def embed(self, sequences):
        sequences = [s.upper().translate(self.aa_map) for s in sequences]

        tokens = [' '.join(list(s)) for s in sequences]
        tokens = self.tokenizer.batch_encode_plus(tokens,
                                                  padding='longest',
                                                  add_special_tokens=True)

        device = self.encoder_model.device
        input_ids = torch.tensor(tokens['input_ids'], device=device)
        attention_mask = torch.tensor(tokens['attention_mask'], device=device)

        embeddings = self.encoder_model(input_ids=input_ids,
                                        attention_mask=attention_mask)

        return embeddings.last_hidden_state
