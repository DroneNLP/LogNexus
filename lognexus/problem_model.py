import json
from pathlib import Path
from typing import Dict, Tuple


def load_droptc(device, model_dir: Path = Path("best-model/droptc")) -> Tuple[object, object, Dict]:
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "torch and transformers are required for DroPTC segment classification."
        ) from exc

    class DroPTC(nn.Module):
        def __init__(
            self,
            embedding_model,
            tokenizer,
            hidden_dim=128,
            dropout_rate=0.1,
            num_class=7,
            freeze_embedding=False,
        ):
            super().__init__()
            self.embedding_model = embedding_model
            self.tokenizer = tokenizer
            self.dense = nn.Linear(embedding_model.config.hidden_size, hidden_dim)
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_class),
            )
            if freeze_embedding:
                for param in self.embedding_model.parameters():
                    param.requires_grad = False

        @staticmethod
        def mean_pooling(last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
            summed = torch.sum(last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts

        def forward(self, input_ids, attention_mask, **kwargs):
            output = self.embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            pooled = self.mean_pooling(output.last_hidden_state, attention_mask)
            hidden = self.dense(pooled)
            return self.classifier(hidden)

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "pytorch_model.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"DroPTC model directory must contain config.json and pytorch_model.pt: {model_dir}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    embedding_model_name = config["embedding_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained(
        embedding_model_name,
        trust_remote_code=True,
    ).to(device)

    model = DroPTC(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        hidden_dim=config.get("hidden_dim", 128),
        dropout_rate=config.get("dropout_rate", 0.1),
        num_class=config.get("num_class", 7),
        freeze_embedding=config.get("freeze_embedding", True),
    ).to(device)
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.eval()
    return model, tokenizer, config
