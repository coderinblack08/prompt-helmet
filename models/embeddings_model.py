from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from datasets import Dataset
from .utils import get_training_and_validation_splits
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from setfit import SetFitModel, TrainingArguments, Trainer
import torch
import numpy as np


class ContrastivePromptEmbeddingTrainer:
    def __init__(self, model_name, total_size=None, load_model=False):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if load_model:
            self.pretrained = True
            self.model = SentenceTransformer("./saved_models/embeddings_model.bin").to(self.device)
        else:
            self.pretrained = False
            self.model = SentenceTransformer(model_name).to(self.device)
        self.trainer = None
        
        # Prepare train and validation datasets during initialization
        (self.train_system_prompts, self.train_user_prompts), (self.val_system_prompts, self.val_user_prompts) = get_training_and_validation_splits(total_size=total_size)
        self.train_dataset = self.prepare_dataset(self.train_system_prompts, self.train_user_prompts)
        self.val_dataset = self.prepare_dataset(self.val_system_prompts, self.val_user_prompts)
        
    def prepare_dataset(self, system_prompts_df, user_prompts_df):
        system_prompts_list = system_prompts_df["system_prompt"].tolist()
        user_prompts_list = user_prompts_df["user_input"].tolist()
        is_injected_list = user_prompts_df["is_injection"].tolist()

        return Dataset.from_dict({
            "sentence1": system_prompts_list,
            "sentence2": user_prompts_list,
            "label": is_injected_list,
        })
        
    def train(self):
        loss = losses.ContrastiveLoss(self.model)
        self.trainer = SentenceTransformerTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            loss=loss
        )
        
        self.trainer.train()
        self.model.save("./saved_models/embeddings_model.bin")
        
    def find_optimal_threshold(self, threshold_range=None):
        if self.trainer is None and not self.pretrained:
            raise ValueError("Model must be trained before finding optimal threshold")

        if threshold_range is None:
            threshold_range = np.arange(-0.2, 1.0, 0.1)
            
        system_prompt_dict = {}
        system_embeddings = []
        for prompt in self.val_system_prompts["system_prompt"].tolist():
            if prompt not in system_prompt_dict:
                system_prompt_dict[prompt] = torch.from_numpy(self.model.encode(prompt))
            system_embeddings.append(system_prompt_dict[prompt])

        system_embeddings = torch.stack(system_embeddings)
        user_embeddings = torch.from_numpy(self.model.encode(self.val_user_prompts["user_input"].tolist()))
        
        system_embeddings = system_embeddings.cpu().numpy()
        user_embeddings = user_embeddings.cpu().numpy()
        
        similarities = cosine_similarity(system_embeddings, user_embeddings)
        true_labels = self.val_user_prompts["is_injection"].tolist()
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in threshold_range:
            predictions = (similarities.diagonal() > threshold).astype(int)
            f1 = f1_score(true_labels, predictions)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1

    def evaluate(self, threshold=None):
        if self.trainer is None and not self.pretrained:
            raise ValueError("Model must be trained before evaluation")
            
        if threshold is None:
            threshold, _ = self.find_optimal_threshold()
            
        system_prompt_dict = {}
        system_embeddings = []
        for prompt in self.val_system_prompts["system_prompt"].tolist():
            if prompt not in system_prompt_dict:
                system_prompt_dict[prompt] = torch.from_numpy(self.model.encode(prompt))
            system_embeddings.append(system_prompt_dict[prompt])

        system_embeddings = torch.stack(system_embeddings)
        user_embeddings = torch.from_numpy(self.model.encode(self.val_user_prompts["user_input"].tolist()))
        
        system_embeddings = system_embeddings.cpu().numpy()
        user_embeddings = user_embeddings.cpu().numpy()
        
        similarities = cosine_similarity(system_embeddings, user_embeddings)
        predictions = (similarities.diagonal() > threshold).astype(int)
        
        accuracy = accuracy_score(self.val_user_prompts["is_injection"].tolist(), predictions)
        precision = precision_score(self.val_user_prompts["is_injection"].tolist(), predictions)
        recall = recall_score(self.val_user_prompts["is_injection"].tolist(), predictions)
        f1 = f1_score(self.val_user_prompts["is_injection"].tolist(), predictions)
        
        results = {
            "num_system_prompts": len(self.val_system_prompts),
            "num_user_prompts": len(self.val_user_prompts),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        
        return results
        
    def get_model(self):
        return self.model


class SetFitPromptEmbeddingTrainer:
    def __init__(self, model_name, total_size=None):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SetFitModel.from_pretrained(model_name).to(self.device)
        self.trainer = None
        
        # Prepare train and validation datasets during initialization
        (self.train_system_prompts, self.train_user_prompts), (self.val_system_prompts, self.val_user_prompts) = get_training_and_validation_splits(total_size=total_size)
        self.train_dataset = self.prepare_dataset(self.train_system_prompts, self.train_user_prompts)
        self.val_dataset = self.prepare_dataset(self.val_system_prompts, self.val_user_prompts)
        
    def prepare_dataset(self, system_prompts_df, user_prompts_df):
        system_prompts_list = system_prompts_df["system_prompt"].tolist()
        user_prompts_list = user_prompts_df["user_input"].tolist()
        is_injected_list = user_prompts_df["is_injection"].tolist()
        
        text_pairs = [
            f"System: {system_prompt} \n User: {user_prompt}"
            for system_prompt, user_prompt in zip(system_prompts_list, user_prompts_list)
        ]
        
        dataset = Dataset.from_dict({
            "text": text_pairs,
            "label": is_injected_list,
        })
        
        return dataset
        
    def train(self, epochs=3, iterations=20):
        args = TrainingArguments(
            num_epochs=epochs,
            num_iterations=iterations,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            metric="f1"
        )
        
        self.trainer.train()
        self.model.save_pretrained("./saved_models/setfit_model")
        
    def evaluate(self, threshold=0.5):
        if self.trainer is None and not self.pretrained:
            raise ValueError("Model must be trained before evaluation")
            
        system_embeddings = self.model.encode(self.val_system_prompts["system_prompt"].tolist())
        user_embeddings = self.model.encode(self.val_user_prompts["user_input"].tolist())
        
        similarities = cosine_similarity(system_embeddings, user_embeddings)
        predictions = (similarities.diagonal() > threshold).astype(int)
        
        accuracy = accuracy_score(self.val_user_prompts["is_injection"].tolist(), predictions)
        precision = precision_score(self.val_user_prompts["is_injection"].tolist(), predictions)
        recall = recall_score(self.val_user_prompts["is_injection"].tolist(), predictions)
        f1 = f1_score(self.val_user_prompts["is_injection"].tolist(), predictions)
        
        results = {
            "num_system_prompts": len(self.val_system_prompts),
            "num_user_prompts": len(self.val_user_prompts),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        
        return results
        
    def get_model(self):
        return self.model
