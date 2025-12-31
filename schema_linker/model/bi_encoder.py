import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import AutoModel, AutoTokenizer


class SchemaLinkingModel(pl.LightningModule):
    SIMILARITY_THRESHOLD = 0.5

    def __init__(self, model_name, learning_rate, margin):
        super().__init__()
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.margin = margin
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.loss_fn = torch.nn.CosineEmbeddingLoss(margin=margin)
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(
            mask.sum(1),
            min=1e-9,
        )

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        questions = batch["question"]
        schema_items = batch["schema_item"]
        labels = batch["label"]

        q_inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        s_inputs = self.tokenizer(
            schema_items,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        q_emb = self(
            q_inputs["input_ids"].to(self.device),
            q_inputs["attention_mask"].to(self.device),
        )
        s_emb = self(
            s_inputs["input_ids"].to(self.device),
            s_inputs["attention_mask"].to(self.device),
        )

        loss = self.loss_fn(q_emb, s_emb, labels.float().to(self.device))
        self.log("train_loss", loss, prog_bar=True)

        sim = torch.nn.functional.cosine_similarity(q_emb, s_emb)
        preds = (sim > self.SIMILARITY_THRESHOLD).long()
        targets_01 = ((labels + 1) // 2).long().to(self.device)
        self.train_f1(preds, targets_01)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        questions = batch["question"]
        schema_items = batch["schema_item"]
        labels = batch["label"]

        q_inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        s_inputs = self.tokenizer(
            schema_items,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        q_emb = self(
            q_inputs["input_ids"].to(self.device),
            q_inputs["attention_mask"].to(self.device),
        )
        s_emb = self(
            s_inputs["input_ids"].to(self.device),
            s_inputs["attention_mask"].to(self.device),
        )

        loss = self.loss_fn(q_emb, s_emb, labels.float().to(self.device))
        self.log("val_loss", loss, prog_bar=True)

        sim = torch.nn.functional.cosine_similarity(q_emb, s_emb)
        preds = (sim > self.SIMILARITY_THRESHOLD).long()
        targets_01 = ((labels + 1) // 2).long().to(self.device)
        self.val_f1(preds, targets_01)
        self.val_recall(preds, targets_01)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def to_onnx(self, file_path: str):
        dummy_question = "What is the average salary?"
        dummy_table = "Table: employee, Columns: id, name, salary"
        inputs = self.tokenizer(
            [dummy_question, dummy_table],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        torch.onnx.export(
            self,
            (input_ids, attention_mask),
            file_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_length"},
                "attention_mask": {0: "batch_size", 1: "seq_length"},
                "embeddings": {0: "batch_size"},
            },
        )
