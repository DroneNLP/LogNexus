import logging
import os
import json

from lognexus.utils import pretokenize_log_message


logging.getLogger("simpletransformers").setLevel(logging.WARNING)


class LogNexusModel:
    def __init__(self, model_dir, use_cuda=False):
        self.model_dir = model_dir
        self.use_cuda = use_cuda
        self.model_type = self._detect_model_type()
        self.model = self._load_model()

    def _detect_model_type(self):
        """
        Reads the 'config.json' in the model directory to find the 'model_type'.
        """
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {self.model_dir}. Cannot detect model type.")

        with open(config_path, 'r') as f:
            config = json.load(f)

        model_type = config.get("model_type")
        if not model_type:
            raise ValueError("Could not find 'model_type' in config.json")

        return model_type

    def _load_model(self):
        try:
            from simpletransformers.ner import NERModel
        except ImportError as exc:
            raise ImportError(
                "simpletransformers is required for inference. "
                "Install LogNexus with its runtime dependencies before loading a model."
            ) from exc

        return NERModel(
            self.model_type,
            self.model_dir,
            use_cuda=self.use_cuda,
            args={"silent": True}
        )

    def predict_sentences(self, messages: list) -> list:
        """
        Performs batch prediction and reconstructs sentences.
        """
        tokenized_messages = [pretokenize_log_message(msg) for msg in messages]
        predictions, _ = self.model.predict(tokenized_messages, split_on_space=False)

        extracted_results = []
        for sentence_preds in predictions:
            extracted_results.append(self._reconstruct_sentences(sentence_preds))

        return extracted_results

    def _reconstruct_sentences(self, token_predictions):
        sentences = []
        current_sent = []
        for i, token_dict in enumerate(token_predictions):
            for word, tag in token_dict.items():
                if tag == 'O' or word == ';':
                    continue
                if tag.startswith("B-") or tag.startswith("S-"):
                    if current_sent:
                        sentences.append(" ".join(current_sent))
                    current_sent = [word]
                    if tag.startswith("S-"):
                        sentences.append(" ".join(current_sent))
                        current_sent = []
                elif tag.startswith("I-") or tag.startswith("E-"):
                    current_sent.append(word)
                    if tag.startswith("E-"):
                        is_last_token = i + 1 >= len(token_predictions)
                        next_is_outside = False
                        if not is_last_token:
                            next_tags = list(token_predictions[i + 1].values())
                            next_is_outside = bool(next_tags and next_tags[0] == 'O')
                        if is_last_token or next_is_outside:
                            sentences.append(" ".join(current_sent))
                            current_sent = []
                else:
                    if current_sent:
                        sentence = " ".join(current_sent)
                        sentences.append(sentence)
                        current_sent = []
        if current_sent:
            sentence = " ".join(current_sent)
            sentences.append(sentence)

        return sentences
