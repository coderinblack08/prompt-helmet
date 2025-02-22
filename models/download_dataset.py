import pandas as pd
import numpy as np
from datasets import load_dataset


dataset = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")
spml_df = pd.DataFrame(dataset['train'])

malicious_examples = spml_df[['Prompt injection', 'Degree', 'User Prompt', 'System Prompt']]
malicious_examples = malicious_examples.rename(columns={
    'Prompt injection': 'is_injection',
    'User Prompt': 'user_input',
    'System Prompt': 'system_prompt'
})
unique_system_prompts = malicious_examples['system_prompt'].unique()
system_prompt_mapping = pd.DataFrame({
    'id': range(1, len(unique_system_prompts) + 1),
    'system_prompt': unique_system_prompts
})
malicious_examples = malicious_examples.merge(
    system_prompt_mapping,
    on='system_prompt',
    how='left'
).drop('system_prompt', axis=1)
malicious_examples = malicious_examples.rename(columns={'id': 'system_prompt_id'})


malicious_examples.to_csv('datasets/user_prompts.csv', index=False)
system_prompt_mapping.to_csv('datasets/system_prompts.csv', index=False)