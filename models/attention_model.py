import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.optim as optim
from .classifiers import SimpleCNNClassifier, RandomForestClassifier, GradientBoostingClassifier
from .utils import get_training_and_validation_splits
import time
import os


class AttentionHeatmapDataset(Dataset):
    def __init__(self, heatmaps, labels):
        self.heatmaps = torch.FloatTensor(np.array(heatmaps))
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.heatmaps)
    
    def __getitem__(self, idx):
        return self.heatmaps[idx], self.labels[idx]


class AttentionModel:
    def __init__(self, model_name, device="mps", classifier_class=SimpleCNNClassifier, total_size=None, batch_size=32, load_model=False, load_dataset=False):
        self.model_name = model_name
        self.device = device
        self.total_size = total_size
        
        # Create safe model name for file paths
        model_name_safe = model_name.replace('/', '_')
        
        # Define dataset and model paths
        self.train_dataset_path = f"datasets/train_attention_dataset_{model_name_safe}.pt"
        self.val_dataset_path = f"datasets/val_attention_dataset_{model_name_safe}.pt"
        self.classifier_path = f"saved_models/attention_classifier_{model_name_safe}.pt"
        
        # Create directories if they don't exist
        os.makedirs("datasets", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)
        
        # Check if we only need to load datasets
        if load_dataset and self._datasets_exist():
            print("Loading saved datasets only...")
            train_dataset = torch.load(self.train_dataset_path)
            val_dataset = torch.load(self.val_dataset_path)
            
            g = torch.Generator(device='cpu')
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
            self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=g)
            
            # Don't load the model or tokenizer
            self.tokenizer = None
            self.model = None
            self.classifier = None
            self.classifier_class = classifier_class
            return
        
        # Load tokenizer and model if we need them
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto"
        ).to(device)
        self.classifier_class = classifier_class
        self.classifier = None

        print("Loaded models")
        
        if load_model and self._saved_files_exist():
            print("Loading saved datasets and model...")
            train_dataset = torch.load(self.train_dataset_path)
            val_dataset = torch.load(self.val_dataset_path)
            
            g = torch.Generator(device='cpu')
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
            self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=g)
            
            sample_heatmap = next(iter(self.train_dataloader))[0][0]
            input_shape = sample_heatmap.shape
            self.classifier = self.classifier_class(input_shape).to(device)
            self.classifier.load_state_dict(torch.load(self.classifier_path))
            self.classifier.eval()
            
        else:
            print("Preparing new datasets and model...")
            (self.train_system_prompts, self.train_user_prompts), (self.val_system_prompts, self.val_user_prompts) = get_training_and_validation_splits(total_size=total_size)
            
            self.train_dataloader = self._prepare_dataloader(self.train_system_prompts, self.train_user_prompts, batch_size)
            self.val_dataloader = self._prepare_dataloader(self.val_system_prompts, self.val_user_prompts, batch_size)

            torch.save(self.train_dataloader.dataset, self.train_dataset_path)
            torch.save(self.val_dataloader.dataset, self.val_dataset_path)
            
            sample_heatmap = next(iter(self.train_dataloader))[0][0]
            input_shape = sample_heatmap.shape
            self.classifier = self.classifier_class(input_shape).to(device)

    def _saved_files_exist(self):
        """Check if all necessary saved files exist."""
        return (os.path.exists(self.train_dataset_path) and 
                os.path.exists(self.val_dataset_path) and 
                os.path.exists(self.classifier_path))
                
    def _datasets_exist(self):
        """Check if dataset files exist."""
        return (os.path.exists(self.train_dataset_path) and 
                os.path.exists(self.val_dataset_path))

    def _prepare_dataloader(self, system_prompts, user_prompts, batch_size=32):
        heatmaps = []
        labels = []
        
        for idx, (sys_prompt, user_prompt) in enumerate(zip(system_prompts["system_prompt"], user_prompts["user_input"])):
            _, _, attention_maps, _, data_range, _ = self.inference(
                instruction=sys_prompt,
                data=user_prompt
            )
            
            heatmap = self.process_attn(attention_maps[0], data_range, "sum")
            heatmaps.append(heatmap)
            
            labels.append(1 if user_prompts["is_injection"][idx] else 0)
            print(f"Processed {idx} out of {len(system_prompts)}")
            
            if idx % 10 == 0 and self.device == "mps":
                torch.mps.empty_cache()
        
        dataset = AttentionHeatmapDataset(heatmaps, labels)
        
        dataset_path = f"datasets/attention_dataset_{len(heatmaps)}_{self.model_name.replace('/', '_')}.pt"
        torch.save(dataset, dataset_path)
        print(f"Dataset saved to {dataset_path}")
        
        g = torch.Generator(device='cpu')
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

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


    def inference(self, instruction, data, max_output_tokens=1):
        if not isinstance(data, str):
            data = str(data)
        
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

    def train(self, epochs=10, learning_rate=1e-4, classifier_kwargs=None):
        """Train the classifier model on the prepared datasets."""
        if isinstance(self.classifier, SimpleCNNClassifier):
            return self._train_cnn(epochs, learning_rate)
        elif isinstance(self.classifier, RandomForestClassifier):
            return self._train_random_forest()
        elif isinstance(self.classifier, GradientBoostingClassifier):
            return self._train_gradient_boosting()
        else:
            raise ValueError(f"Unsupported classifier type: {type(self.classifier).__name__}")
    
    def _train_cnn(self, epochs=10, learning_rate=1e-4):
        """Train a CNN-based classifier."""
        device = torch.device(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.classifier.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        
        best_f1 = 0
        
        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            for heatmaps, labels in self.train_dataloader:
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
            
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training Loss: {total_loss/len(self.train_dataloader):.4f}")
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Training F1: {train_f1:.4f}")
            
            val_metrics = self.evaluate()
            print(f"Validation Metrics: {val_metrics}")
            
            scheduler.step(val_metrics['f1'])
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(self.classifier.state_dict(), self.classifier_path)
        
        return val_metrics
    
    def _train_random_forest(self):
        """Train a Random Forest classifier."""
        return self._train_sklearn_model("Random Forest")
    
    def _train_gradient_boosting(self):
        """Train a Gradient Boosting classifier."""
        return self._train_sklearn_model("Gradient Boosting")
    
    def _train_sklearn_model(self, model_name):
        """Generic training function for sklearn-based models."""
        device = torch.device(self.device)
        print(f"Training {model_name} classifier...")
        
        all_train_heatmaps = []
        all_train_labels = []
        
        for heatmaps, labels in self.train_dataloader:
            all_train_heatmaps.append(heatmaps)
            all_train_labels.append(labels)
        
        train_heatmaps = torch.cat(all_train_heatmaps, dim=0)
        train_labels = torch.cat(all_train_labels, dim=0)
        
        self.classifier.fit(train_heatmaps, train_labels)
        
        train_outputs = self.classifier(train_heatmaps)
        train_preds = torch.argmax(train_outputs, dim=1).cpu().numpy()
        train_labels = train_labels.numpy()
        
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Training F1: {train_f1:.4f}")
        
        val_metrics = self.evaluate()
        print(f"Validation Metrics: {val_metrics}")
        
        torch.save(self.classifier.state_dict(), self.classifier_path)
        
        return val_metrics

    def evaluate(self):
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Please train the model first.")
            
        device = torch.device(self.device)
        self.classifier.eval()
        
        all_preds = []
        all_labels = []
        all_times = []
        all_probs = []
        
        with torch.no_grad():
            for heatmaps, labels in self.val_dataloader:
                for i in range(len(heatmaps)):
                    single_heatmap = heatmaps[i:i+1].to(device)
                    
                    start_time = time.time()
                    outputs = self.classifier(single_heatmap)
                    end_time = time.time()
                    
                    inference_time_ms = (end_time - start_time) * 1000
                    all_times.append(inference_time_ms)
                    
                    probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                    pred = np.argmax(probs)
                    all_probs.append(probs[1])  # Probability of positive class
                    all_preds.append(pred)
                    all_labels.append(labels[i].item())
        
        avg_time_per_prompt_ms = sum(all_times) / len(all_times)
        
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'auc_roc': roc_auc_score(all_labels, all_probs),
            'avg_time_per_prompt_ms': avg_time_per_prompt_ms,
            'min_time_per_prompt_ms': min(all_times),
            'max_time_per_prompt_ms': max(all_times)
        }

    def predict(self, heatmap):
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Please train the model first.")
            
        device = torch.device(self.device)
        self.classifier.eval()
        
        with torch.no_grad():
            heatmap_tensor = torch.FloatTensor(heatmap).unsqueeze(0).to(device)
            
            start_time = time.time()
            output = self.classifier(heatmap_tensor)
            end_time = time.time()
            
            pred = torch.argmax(output, dim=1).item()
            prob = F.softmax(output, dim=1)[0][pred].item()
            
            inference_time_ms = (end_time - start_time) * 1000
        
        return pred, prob, inference_time_ms

    def load_classifier(self, state_dict_path):
        """Load a trained classifier from a state dict file."""
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Please initialize the model with proper input shape first.")
        
        self.classifier.load_state_dict(torch.load(state_dict_path))
        self.classifier.eval()
