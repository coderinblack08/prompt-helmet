import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from .classifiers import CNNTransformerClassifier


class AttentionHeatmapDataset(Dataset):
    def __init__(self, heatmaps, labels):
        self.heatmaps = torch.FloatTensor(np.array(heatmaps))
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.heatmaps)
    
    def __getitem__(self, idx):
        return self.heatmaps[idx], self.labels[idx]


class AttentionModel:
    def __init__(self, model_name, device="mps", classifier_class=CNNTransformerClassifier):
        torch.set_default_device(device)
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype="auto",
            device_map="auto"
        )
        self.classifier_class = classifier_class
        self.classifier = None


    def get_last_attn(self, attn_map):
        for i, layer in enumerate(attn_map):
            attn_map[i] = layer[:, :, -1, :].unsqueeze(2)
        return attn_map


    def sample_token(self, logits, top_k=None, top_p=None, temperature=1.0):
        logits = logits / temperature

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k)
            probs = F.softmax(values, dim=-1)
            next_token_id = indices[torch.multinomial(probs, 1)]
            return next_token_id

        return logits.argmax(dim=-1).squeeze()


    def inference(self, instruction, data, max_output_tokens=None):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "Data: " + data}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        instruction_len = len(self.tokenizer.encode(instruction))
        data_len = len(self.tokenizer.encode(data))

        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.model.device)
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            model_inputs['input_ids'][0])

        if "Qwen" in self.model_name:
            data_range = ((3, 3+instruction_len), (-5-data_len, -5))
        elif "phi3" in self.model_name:
            data_range = ((1, 1+instruction_len), (-2-data_len, -2))
        elif "llama3-8b" in self.model_name:
            data_range = ((5, 5+instruction_len), (-5-data_len, -5))
        elif "mistral-7b" in self.model_name:
            data_range = ((3, 3+instruction_len), (-1-data_len, -1))
        else:
            raise NotImplementedError

        generated_tokens = []
        generated_probs = []
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        attention_maps = []

        if max_output_tokens is None:
            n_tokens = 100
        else:
            n_tokens = max_output_tokens

        with torch.no_grad():
            for i in range(n_tokens):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )

                logits = output.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token_id = self.sample_token(
                    logits[0], top_k=50, top_p=None, temperature=1.0)[0]

                generated_probs.append(probs[0, next_token_id.item()].item())
                generated_tokens.append(next_token_id.item())

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat(
                    (input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1)
                attention_mask = torch.cat(
                    (attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=-1)

                attention_map = [attention.detach().cpu().half()
                                 for attention in output['attentions']]
                attention_map = [torch.nan_to_num(
                    attention, nan=0.0) for attention in attention_map]
                attention_map = self.get_last_attn(attention_map)
                attention_maps.append(attention_map)

        output_tokens = [self.tokenizer.decode(
            token, skip_special_tokens=True) for token in generated_tokens]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        return generated_text, output_tokens, attention_maps, input_tokens, data_range, generated_probs


    def process_attn(self, attention, rng, attn_func):
        heatmap = np.zeros((len(attention), attention[0].shape[1]))
        for i, attn_layer in enumerate(attention):
            attn_layer = attn_layer.to(torch.float32).numpy()

            if "sum" in attn_func:
                last_token_attn_to_inst = np.sum(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
                attn = last_token_attn_to_inst
            
            elif "max" in attn_func:
                last_token_attn_to_inst = np.max(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
                attn = last_token_attn_to_inst
            else:
                raise NotImplementedError
                
            last_token_attn_to_inst_sum = np.sum(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
            last_token_attn_to_data_sum = np.sum(attn_layer[0, :, -1, rng[1][0]:rng[1][1]], axis=1)

            if "normalize" in attn_func:
                epsilon = 1e-8
                heatmap[i, :] = attn / (last_token_attn_to_inst_sum + last_token_attn_to_data_sum + epsilon)
            else:
                heatmap[i, :] = attn

        heatmap = np.nan_to_num(heatmap, nan=0.0)
        return heatmap


    def calc_attn_score(self, heatmap):
        return np.sum(heatmap)


    def prepare_training_data(self, system_prompts, user_prompts, batch_size=32):
        heatmaps = []
        labels = []
        
        for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
            # Generate attention maps through inference
            _, _, attention_maps, _, data_range, _ = self.inference(
                instruction=sys_prompt,
                data=user_prompt
            )
            
            # Process attention maps to create heatmap
            heatmap = self.process_attn(attention_maps[0], data_range, "sum")
            heatmaps.append(heatmap)
            
            # Assuming user_prompts has an 'is_injection' column
            labels.append(1 if user_prompt.is_injection else 0)
        
        # Create dataset and dataloader
        dataset = AttentionHeatmapDataset(heatmaps, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, train_dataloader, val_dataloader=None, epochs=10, learning_rate=1e-4, classifier_kwargs=None):
        device = torch.device(self.device)
        
        # Initialize classifier if not already initialized
        if self.classifier is None:
            sample_heatmap = next(iter(train_dataloader))[0][0]
            input_shape = sample_heatmap.shape
            classifier_kwargs = classifier_kwargs or {}
            self.classifier = self.classifier_class(input_shape, **classifier_kwargs).to(device)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.classifier.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        
        best_f1 = 0
        save_path = f'best_{self.classifier.get_name()}_classifier.pt'
        
        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            for heatmaps, labels in train_dataloader:
                heatmaps, labels = heatmaps.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.classifier(heatmaps)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            
            # Calculate training metrics
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training Loss: {total_loss/len(train_dataloader):.4f}")
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Training F1: {train_f1:.4f}")
            
            # Validation
            if val_dataloader:
                val_metrics = self.evaluate(val_dataloader)
                print(f"Validation Metrics: {val_metrics}")
                
                # Learning rate scheduling based on F1 score
                scheduler.step(val_metrics['f1'])
                
                # Save best model
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    torch.save(self.classifier.state_dict(), save_path)

    def evaluate(self, dataloader):
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Please train the model first.")
            
        device = torch.device(self.device)
        self.classifier.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for heatmaps, labels in dataloader:
                heatmaps = heatmaps.to(device)
                outputs = self.classifier(heatmaps)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds)
        }

    def predict(self, heatmap):
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Please train the model first.")
            
        device = torch.device(self.device)
        self.classifier.eval()
        
        with torch.no_grad():
            heatmap_tensor = torch.FloatTensor(heatmap).unsqueeze(0).to(device)
            output = self.classifier(heatmap_tensor)
            pred = torch.argmax(output, dim=1).item()
            prob = F.softmax(output, dim=1)[0][pred].item()
        
        return pred, prob

    def load_classifier(self, state_dict_path):
        """Load a trained classifier from a state dict file."""
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Please initialize the model with proper input shape first.")
        
        self.classifier.load_state_dict(torch.load(state_dict_path))
        self.classifier.eval()
