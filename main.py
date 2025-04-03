from models.attention_model import AttentionModel
from models.embeddings_model import ContrastivePromptEmbeddingTrainer
from models.classifiers import SimpleCNNClassifier


def run_embedding_model(model_name, total_size, train):
    if train:
        contrastive_trainer = ContrastivePromptEmbeddingTrainer(model_name=model_name, total_size=total_size, load_model=False)
        contrastive_trainer.train()
        results = contrastive_trainer.evaluate()
        return results
    else:
        contrastive_trainer = ContrastivePromptEmbeddingTrainer(model_name=model_name, total_size=total_size, load_model=True)
        results = contrastive_trainer.evaluate()
        return results


def run_attention_model(model_name, total_size, classifier_class, train):
    if train:
        attention_model = AttentionModel(model_name=model_name, classifier_class=classifier_class, total_size=total_size, load_model=False)
        attention_model.train()
        results = attention_model.evaluate()
        return results
    else:
        attention_model = AttentionModel(model_name=model_name, classifier_class=classifier_class, total_size=total_size, load_model=True)
        results = attention_model.evaluate()
        return results


def main():
    print(run_embedding_model(model_name="all-MiniLM-L6-v2", total_size=None, train=False))
    print(run_embedding_model(model_name="all-mpnet-base-v2", total_size=None, train=False))

    print(run_attention_model(classifier_class=SimpleCNNClassifier(), model_name="Qwen/Qwen2.5-1.5B-Instruct", train=True, total_size=100))


if __name__ == "__main__":
    main()
