import wandb
import pandas as pd

api = wandb.Api()

project = "kevinlu-email-bellarmine-college-preparatory/sentence-transformers"
runs = api.runs(project)

for run in runs:
    history = run.history()
    
    if 'train/loss' in history.columns and 'train/grad_norm' in history.columns:
        for i, row in history.iterrows():
            if pd.isna(row.get('train/loss')):
                continue
                
            metrics = {
                'loss': round(float(row['train/loss']), 4),
                'grad_norm': float(row['train/grad_norm']),
                'learning_rate': float(row['train/learning_rate']),
                'epoch': round(float(row['train/epoch']), 2)
            }
            print(metrics)
        break
