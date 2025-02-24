from models.attention_model import AttentionModel
from models.embeddings_model import ContrastivePromptEmbeddingTrainer
from models.classifiers import SimpleCNNClassifier


def main():
    ## Embedding Model:
    # contrastive_trainer = ContrastivePromptEmbeddingTrainer(model_name="all-MiniLM-L6-v2", total_size=200, load_model=True)
    # contrastive_trainer.train()
    # results = contrastive_trainer.evaluate()
    # print(results)

    ## Attention Model:
    attention_model = AttentionModel(model_name="Qwen/Qwen2.5-1.5B-Instruct", classifier_class=SimpleCNNClassifier, total_size=50)
    attention_model.train()
    results = attention_model.evaluate()
    print(results)


if __name__ == "__main__":
    main()
