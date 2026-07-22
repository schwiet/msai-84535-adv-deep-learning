import abc

import torch
from torch.nn.modules.transformer import _generate_square_subsequent_mask


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(
        self, B: int = 1, h: int = 20, w: int = 30, device=None
    ) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.embed = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_latent)
        transfLayer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.transf = torch.nn.TransformerEncoder(
            encoder_layer=transfLayer, num_layers=4
        )
        self.register_buffer(
            "mask", torch.nn.Transformer.generate_square_subsequent_mask(600)
        )
        self.score = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        emb = self.embed(torch.flatten(x, 1))
        padding = torch.zeros_like(emb[:, :1, :])
        shifted = torch.cat([padding, emb[:, :-1, :]], dim=1)

        return torch.unflatten(
            self.score(self.transf(shifted, mask=self.mask)),
            1,
            (x.shape[1], x.shape[2]),
        ), {}

    def generate(
        self, B: int = 1, h: int = 30, w: int = 20, device=None
    ) -> torch.Tensor:  # noqa
        with torch.no_grad():
            canvas = torch.zeros((B, h, w), dtype=torch.long, device=device)
            # Loop over grid positions and write the sampled tokens into the canvas at (i, j)
            for i in range(h):
                for j in range(w):
                    pred, _ = self.forward(canvas)
                    logits = pred[:, i, j, :]
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1)
                    canvas[:, i, j] = sampled[:, 0]
            return canvas
