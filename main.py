from models.embeddings_model import ContrastivePromptEmbeddingTrainer


def main():
    contrastive_trainer = ContrastivePromptEmbeddingTrainer(model_name="all-MiniLM-L6-v2", total_size=200, load_model=True)
    # contrastive_trainer.train()
    results = contrastive_trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()
