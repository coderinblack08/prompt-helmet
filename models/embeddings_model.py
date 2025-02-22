from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from datasets import Dataset
from .utils import get_training_and_validation_splits
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PromptEmbeddingTrainer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.trainer = None
        
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
        (train_system_prompts, train_user_prompts), (_, _) = get_training_and_validation_splits()
        train_dataset = self.prepare_dataset(train_system_prompts, train_user_prompts)
        
        loss = losses.ContrastiveLoss(self.model)
        self.trainer = SentenceTransformerTrainer(
            model=self.model,
            train_dataset=train_dataset,
            loss=loss
        )
        
        self.trainer.train()
        self.model.save("./saved_models/embeddings_model.bin")
        
    def evaluate(self, threshold=0.5):
        (_, _), (val_system_prompts, val_user_prompts) = get_training_and_validation_splits()
        system_embeddings = self.model.encode(val_system_prompts["system_prompt"].tolist())
        user_embeddings = self.model.encode(val_user_prompts["user_input"].tolist())
        
        similarities = cosine_similarity(system_embeddings, user_embeddings)
        predictions = (similarities.diagonal() > threshold).astype(int)
        
        accuracy = accuracy_score(val_user_prompts["is_injection"].tolist(), predictions)
        precision = precision_score(val_user_prompts["is_injection"].tolist(), predictions)
        recall = recall_score(val_user_prompts["is_injection"].tolist(), predictions)
        f1 = f1_score(val_user_prompts["is_injection"].tolist(), predictions)
        
        results = {
            "num_system_prompts": len(val_system_prompts),
            "num_user_prompts": len(val_user_prompts),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        
        return results
        
    def get_model(self):
        return self.model
