from pathlib import Path


class DonorPredictor:
    def __init__(self, *, encoder_path: Path, readout_path: Path) -> None:
        import joblib
        import numpy as np
        import torch

        from gtb_predict.compact import CompactProtein
        from gtb_predict.data import DONORS

        torch.set_num_threads(2)
        saved = torch.load(encoder_path, map_location="cpu", weights_only=True)
        if (saved["width"], saved["layers"], saved["input_tokens"]) != (
            512,
            16,
            100043076,
        ):
            raise ValueError("Unexpected donor encoder configuration")
        self.encoder = CompactProtein(512, 16).eval().requires_grad_(False)
        loaded = self.encoder.load_state_dict(saved["encoder"], strict=False)
        auxiliary = ("classifier.", "reconstruct.", "coherence.")
        if loaded.unexpected_keys or any(
            not name.startswith(auxiliary) for name in loaded.missing_keys
        ):
            raise ValueError("Incomplete donor residue encoder")
        readout = joblib.load(readout_path)
        self.readout = readout["model"]
        svc = self.readout.named_steps["classifier"]
        if (
            list(self.readout.named_steps) != ["scale", "normalize", "classifier"]
            or svc.probability
            or svc.C != 100
            or svc.kernel != "rbf"
            or not np.array_equal(self.readout.classes_, np.arange(8))
            or self.readout.named_steps["scale"].n_samples_seen_ != 413
        ):
            raise ValueError("Unexpected frozen donor readout")
        self.donors = tuple(DONORS)
        if len(self.donors) != 8 or self.donors[7] != "Eight":
            raise ValueError("Unexpected donor class order")

    def _encode(self, sequence: str):
        import numpy as np
        import torch

        from gtb_predict.compact import token_batch

        if not isinstance(sequence, str) or not 319 <= len(sequence) <= 2358:
            raise ValueError("Donor prediction accepts 319-2358 residues")
        if set(sequence) - set("ACDEFGHIKLMNPQRSTVWYBXZJUO*"):
            raise ValueError("Donor prediction requires uppercase amino-acid symbols")
        with torch.inference_mode():
            tokens = token_batch([sequence], device="cpu")
            hidden, valid = self.encoder.residues(tokens)
            encoded = self.encoder.representation(hidden, valid).float().numpy()
        if encoded.shape != (1, 2048) or not np.isfinite(encoded).all():
            raise ValueError("Invalid donor representation")
        return encoded

    def predict(self, sequence: str) -> dict:
        import numpy as np
        from sklearn.preprocessing import normalize

        normalized = normalize(self._encode(sequence))
        prediction = self.readout.predict(normalized)
        margins = self.readout.decision_function(normalized)
        if (
            prediction.shape != (1,)
            or margins.shape != (1, 8)
            or not np.isfinite(margins).all()
        ):
            raise ValueError("Invalid donor classification")
        index = int(prediction[0])
        if index not in range(len(self.donors)):
            raise ValueError("Unrecognized donor class")
        return {
            "class_index": index,
            "source_label": self.donors[index],
            "sequence_length": len(sequence),
            "ambiguous_residues": sum(sequence.count(symbol) for symbol in "BXZJUO*"),
            "mode": "cpu_fp32",
        }
