import pytest
import torch

from schema_linker.model.bi_encoder import SchemaLinkingModel


class TestSchemaLinkingModel:
    @pytest.fixture()
    def model(self):
        return SchemaLinkingModel(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            learning_rate=1e-5,
            margin=0.5,
        )

    def test_initialization(self, model):
        assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert model.learning_rate == 1e-5
        assert model.margin == 0.5
        assert hasattr(model, "model")
        assert hasattr(model, "tokenizer")
        assert hasattr(model, "loss_fn")
        assert hasattr(model, "train_f1")
        assert hasattr(model, "val_f1")
        assert hasattr(model, "val_recall")

    def test_forward_pass(self, model):
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        embeddings = model(input_ids, attention_mask)
        assert embeddings.shape == (2, 384)

    def test_training_step(self, model):
        batch = {
            "question": ["What is the salary?", "How many employees?"],
            "schema_item": ["employee.salary", "employee.count"],
            "label": torch.tensor([1, -1]),
        }
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_validation_step(self, model):
        batch = {
            "question": ["What is the salary?", "How many employees?"],
            "schema_item": ["employee.salary", "employee.count"],
            "label": torch.tensor([1, -1]),
        }
        loss = model.validation_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_configure_optimizers(self, model):
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-5

    def test_to_onnx(self, model, tmp_path):
        file_path = tmp_path / "test_model.onnx"
        model.to_onnx(str(file_path))
        assert file_path.exists()
