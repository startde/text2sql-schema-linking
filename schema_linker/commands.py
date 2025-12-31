import logging
import shutil
import zipfile
from pathlib import Path

import fire
import gdown
import hydra
import pytorch_lightning as pl
from datasets import load_dataset
from hydra import utils as hydra_utils


class CLI:
    def download(self):
        """Download Spider data from Google Drive and load dataset"""
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        zip_path = raw_dir / "spider_data.zip"
        gdown.download(
            id="1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J",
            output=str(zip_path),
            quiet=False,
        )

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

        tables_src = raw_dir / "spider_data" / "tables.json"
        if tables_src.exists():
            shutil.move(str(tables_src), str(raw_dir / "tables.json"))
            shutil.rmtree(raw_dir / "spider_data")

        zip_path.unlink()

        macosx = raw_dir / "__MACOSX"
        if macosx.exists():
            shutil.rmtree(macosx)

        dataset = load_dataset("spider")
        logging.info("Dataset loaded: %s", dataset)

    def preprocess(self):
        """Preprocess data to create training pairs"""
        tables_path = Path("data/raw/tables.json")
        if not tables_path.exists():
            logging.info("Tables.json not found. Starting download...")
            self.download()

        from schema_linker.data.parser import preprocess_data

        preprocess_data()

    def train(self, config_name="config"):
        """Train the model"""
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(config_name=config_name)

        data_path = Path(hydra_utils.to_absolute_path(cfg.data.train_path))
        if not data_path.exists():
            logging.info(
                "Dataset not found at %s. Starting automatic preprocessing...",
                data_path,
            )
            self.preprocess()
            logging.info("Preprocessing done. Starting training...")

        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import MLFlowLogger

        from schema_linker.data.dataset import SpiderDataModule
        from schema_linker.model.bi_encoder import SchemaLinkingModel

        model = SchemaLinkingModel(
            model_name=cfg.model.model_name,
            learning_rate=cfg.model.learning_rate,
            margin=cfg.model.margin,
        )

        datamodule = SpiderDataModule(
            train_path=cfg.data.train_path,
            val_path=cfg.data.val_path,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )

        logger = MLFlowLogger(
            tracking_uri=cfg.train.mlflow_uri,
            experiment_name="schema-linking-exp",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath="models/",
            filename="best-checkpoint",
        )

        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            accelerator=cfg.train.accelerator,
            logger=logger,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(model, datamodule)

    def infer(self, checkpoint_path, question, candidates):
        """Run inference on question with candidates"""
        import torch

        from schema_linker.model.bi_encoder import SchemaLinkingModel

        model = SchemaLinkingModel.load_from_checkpoint(checkpoint_path)
        model.eval()
        model.freeze()

        q_inputs = model.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        c_inputs = model.tokenizer(
            candidates,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            q_emb = model(q_inputs["input_ids"], q_inputs["attention_mask"])
            c_emb = model(c_inputs["input_ids"], c_inputs["attention_mask"])

        sims = torch.nn.functional.cosine_similarity(q_emb, c_emb, dim=1)

        scores, indices = sims.sort(descending=True)

        logging.info("Results:")
        for i, idx in enumerate(indices):
            logging.info("%s | %.4f", candidates[idx], scores[i].item())

    def export(self, checkpoint_path, output_path="model.onnx"):
        """Export model to ONNX"""
        from schema_linker.model.bi_encoder import SchemaLinkingModel

        model = SchemaLinkingModel.load_from_checkpoint(checkpoint_path)
        model.to_onnx(output_path)
        logging.info("Model exported to %s", output_path)


if __name__ == "__main__":
    fire.Fire(CLI)
