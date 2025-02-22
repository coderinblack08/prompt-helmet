import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=1)
def pull_datasets():
    user_prompts = pd.read_csv("./datasets/user_prompts.csv")
    system_prompts = pd.read_csv("./datasets/system_prompts.csv")

    return user_prompts, system_prompts


def get_all_user_prompts(system_prompt_id):
    user_prompts, _ = pull_datasets()
    return user_prompts[user_prompts["system_prompt_id"] == system_prompt_id]


def get_training_and_validation_splits(split_ratio=0.8, total_size=None):
    _, system_prompts = pull_datasets()
    
    if total_size is not None:
        system_prompts = system_prompts.sample(n=min(total_size, len(system_prompts)), random_state=42)
    
    train_system_prompts = system_prompts.sample(frac=split_ratio, random_state=42)
    val_system_prompts = system_prompts[~system_prompts.index.isin(train_system_prompts.index)]
    
    user_prompts_cache = {}
    for system_prompt_id in system_prompts.index:
        user_prompts_cache[system_prompt_id] = get_all_user_prompts(system_prompt_id)
    
    train_user_prompts = pd.concat([
        user_prompts_cache[system_prompt_id]
        for system_prompt_id in train_system_prompts.index
    ])
    
    val_user_prompts = pd.concat([
        user_prompts_cache[system_prompt_id]
        for system_prompt_id in val_system_prompts.index
    ])
    
    train_system_prompts_expanded = pd.concat([
        pd.DataFrame([train_system_prompts.loc[system_prompt_id]] * len(user_prompts_cache[system_prompt_id]))
        for system_prompt_id in train_system_prompts.index
    ]).reset_index(drop=True)
    
    val_system_prompts_expanded = pd.concat([
        pd.DataFrame([val_system_prompts.loc[system_prompt_id]] * len(user_prompts_cache[system_prompt_id]))
        for system_prompt_id in val_system_prompts.index
    ]).reset_index(drop=True)
    
    train_user_prompts = train_user_prompts.reset_index(drop=True)
    val_user_prompts = val_user_prompts.reset_index(drop=True)
    
    return (train_system_prompts_expanded, train_user_prompts), (val_system_prompts_expanded, val_user_prompts)