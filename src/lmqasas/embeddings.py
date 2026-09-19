"""CPU inference using a verified local AbLang2 model; no model downloads."""
from pathlib import Path
from time import perf_counter
import sys

from .checkpoint import verify_model
from .selection import CANONICAL_AMINO_ACIDS


def reject_network(event: str, args: tuple) -> None:
    if event in {'socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.sendto'}:
        raise RuntimeError('Network access is disabled during local analysis.')


def enable_network_guard() -> None:
    """One-way process guard; CLI uses a separate Python process per analysis."""
    sys.addaudithook(reject_network)


class LocalAbLang2:
    def __init__(self, model_dir: Path, *, batch_size: int = 32, threads: int = 4,
                 seed: int = 20260919):
        import ablang2
        import torch
        for value in (batch_size, threads):
            if type(value) is not int or value < 1:
                raise ValueError('Batch size and threads must be positive integers.')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Seed must be an integer in [0, 2**32).')
        folder = model_dir.resolve(strict=True)
        if 'ABLANG-' not in str(folder):
            raise ValueError('Local AbLang2 path must contain ABLANG-.')
        manifest = verify_model(folder)
        if manifest['hparams'].get('hidden_embed_size') != 480:
            raise ValueError('Expected a 480-dimensional checkpoint.')
        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        self.model = ablang2.pretrained(model_to_use=str(folder), device='cpu', ncpu=1)
        self.model.freeze()
        self.batch_size = batch_size
        self.metadata = {
            'model': manifest, 'device': 'cpu', 'batch_size': batch_size,
            'torch_threads': threads, 'seed': seed,
            'model_parameter_dtype': str(next(self.model.AbLang.parameters()).dtype),
            'mode': 'official_seqcoding', 'align': False, 'fragmented': False,
            'input_format': '[[heavy_CDR3, empty_light_chain], ...]',
            'pooling': 'Mean of all non-padding formatted tokens including <, > and |.',
            'paper_pooling_equivalence': 'unconfirmed',
            'retraining': False,
        }

    def encode(self, sequences: list[str], progress=None):
        import numpy as np
        import torch
        if not sequences or any(not isinstance(s, str) or not s or
                                not set(s).issubset(CANONICAL_AMINO_ACIDS) for s in sequences):
            raise ValueError('Expected nonempty canonical amino-acid strings.')
        started = perf_counter()
        result = np.empty((len(sequences), 480), dtype=np.float64)
        # Length batching reduces padding, but restores the supplied sequence order.
        order = sorted(range(len(sequences)), key=lambda i: (len(sequences[i]), sequences[i]))
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                indexes = order[start:start + self.batch_size]
                pairs = [[sequences[i], ''] for i in indexes]
                batch = self.model(pairs, mode='seqcoding', align=False,
                                   fragmented=False, batch_size=self.batch_size)
                if batch.shape != (len(indexes), 480) or not np.isfinite(batch).all():
                    raise RuntimeError('AbLang2 produced invalid embeddings.')
                result[indexes] = batch
                if progress:
                    progress('embedding', (start + len(indexes)) / len(order))
        self.metadata.update({'embedding_dtype': str(result.dtype),
                              'embedding_seconds': perf_counter() - started,
                              'batch_order': 'length_then_sequence_restored_to_input_order'})
        return result
