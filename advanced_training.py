import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress NUMA warnings
os.environ['TF_DISABLE_NUMA_WARNING'] = '1'

# Import absl before TensorFlow
from absl import logging
logging.set_verbosity(logging.ERROR)

import tensorflow as tf
# Disable TensorFlow logging
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set memory limit to avoid OOM errors
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]  # 10GB limit
        )
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

import numpy as np
from model.transformer_model import DakitariInstructModel
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from data.medical_conversation import MedicalConversationProcessor
from safetensors.tensorflow import load_file, save_file
import shutil

class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_best_only=False, monitor='loss'):
        super().__init__()
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best = float('inf') if 'loss' in monitor else float('-inf')
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.save_best_only:
            if ('loss' in self.monitor and current < self.best) or \
               ('loss' not in self.monitor and current > self.best):
                self.best = current
                print(f"\nSaving best model checkpoint...")
                # Use TensorFlow's native checkpoint system
                checkpoint = tf.train.Checkpoint(model=self.model)
                checkpoint.save(os.path.join(self.filepath, "checkpoint"))

class AdvancedTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def _get_medical_terms_mask(self, y_true):
        """Convert to float32 to match the mask type"""
        return tf.cast(y_true, tf.float32)
    
    def supervised_training(self, train_dataset, output_dir, learning_rate=1e-5):
        # Create TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': train_dataset['input_ids'],
                'attention_mask': train_dataset['attention_mask']
            },
            train_dataset['labels']  # Target data
        )).batch(16)

        def medical_aware_loss(y_true, y_pred):
            # Convert mask to float32
            mask = tf.cast(y_true != -100, tf.float32)
            y_true_processed = tf.where(y_true == -100, tf.zeros_like(y_true), y_true)
            y_true_processed = tf.cast(y_true_processed, tf.float32)
            
            # Ensure y_pred is float32
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Calculate base loss
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true_processed, y_pred, from_logits=True)  # Changed to True since we're using logits
            
            # Get medical terms mask and ensure float32
            medical_terms_mask = tf.cast(self._get_medical_terms_mask(y_true_processed), tf.float32) * mask
            
            # Apply medical terms weighting
            weighted_loss = ce_loss * (1.0 + medical_terms_mask)
            
            return tf.reduce_mean(weighted_loss)
        
        # Configure optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=learning_rate,
                first_decay_steps=1000
            ),
            clipnorm=1.0
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=medical_aware_loss,
            metrics=['accuracy']
        )
        
        # Create checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir,
            save_best_only=True,
            monitor='loss'
        )
        
        # Train the model
        return self.model.fit(
            train_dataset,
            epochs=3,
            callbacks=[checkpoint_callback]
        )
    
    def self_supervised_learning(self, unlabeled_texts, batch_size=8):
        """Self-supervised learning using masked language modeling"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        
        @tf.function
        def train_step(inputs):
            with tf.GradientTape() as tape:
                # Randomly mask 15% of input tokens
                masked_inputs, mask_positions = self._create_mlm_inputs(inputs)
                predictions = self.model(masked_inputs, training=True)
                loss = self._compute_mlm_loss(predictions, inputs, mask_positions)
                
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss
            
        # Training loop
        for epoch in range(3):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(unlabeled_texts), batch_size):
                batch_texts = unlabeled_texts[i:i + batch_size]
                inputs = self.tokenizer(batch_texts, padding=True, return_tensors="tf")
                loss = train_step(inputs)
                total_loss += loss
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}, Average MLM Loss: {avg_loss:.4f}")
    
    def reinforcement_learning(self, eval_dataset, num_episodes=1000):
        """Reinforcement learning using policy gradient"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        
        @tf.function
        def rl_step(state, action_probs):
            with tf.GradientTape() as tape:
                # Generate response
                outputs = self.model(state, training=True)
                
                # Compute reward based on medical accuracy and response quality
                reward = self._compute_medical_reward(outputs, action_probs)
                
                # Policy gradient loss
                loss = -tf.reduce_mean(tf.math.log(action_probs) * reward)
                
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss, reward
        
        # Training loop
        for episode in tqdm(range(num_episodes)):
            for batch in eval_dataset:
                state = batch['input_ids']
                action_probs = tf.nn.softmax(self.model(state))
                loss, reward = rl_step(state, action_probs)
                
                if episode % 100 == 0:
                    print(f"Episode {episode}, Loss: {loss:.4f}, Reward: {reward:.4f}")
    
    def _create_mlm_inputs(self, inputs):
        """Create masked inputs for MLM training"""
        mask_token_id = self.tokenizer.mask_token_id
        input_ids = inputs['input_ids']
        
        # Create random mask
        prob_matrix = tf.random.uniform(shape=tf.shape(input_ids))
        mask = prob_matrix < 0.15
        
        # Replace masked tokens with [MASK]
        masked_inputs = tf.where(mask, mask_token_id, input_ids)
        
        return {'input_ids': masked_inputs}, mask
    
    def _compute_mlm_loss(self, predictions, original_inputs, mask_positions):
        """Compute MLM loss"""
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            original_inputs['input_ids'],
            predictions,
            from_logits=True
        )
        masked_loss = loss * tf.cast(mask_positions, tf.float32)
        return tf.reduce_mean(masked_loss)
    
    def _compute_medical_reward(self, outputs, action_probs):
        """Compute reward for reinforcement learning"""
        # Add medical accuracy reward
        medical_accuracy = self._evaluate_medical_accuracy(outputs)
        
        # Add response coherence reward
        coherence_score = self._evaluate_coherence(outputs)
        
        # Combine rewards
        return medical_accuracy * 0.7 + coherence_score * 0.3
    
    def _evaluate_medical_accuracy(self, outputs):
        """Evaluate medical accuracy of the response"""
        # Add your medical accuracy evaluation logic here
        return tf.random.uniform(shape=(), minval=0, maxval=1)
    
    def _evaluate_coherence(self, outputs):
        """Evaluate response coherence"""
        # Add your coherence evaluation logic here
        return tf.random.uniform(shape=(), minval=0, maxval=1)

def load_model(model_dir):
    """Load the model and tokenizer"""
    print("Loading model...")
    
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load weights first to check shapes
    weights_path = os.path.join(model_dir, "model.safetensors")
    try:
        weights = load_file(weights_path)
        print(f"Successfully loaded weights from {weights_path}")
        
        # Analyze weight shapes to determine correct dimensions
        print("\nAnalyzing weight shapes...")
        weight_shapes = {}
        moe_layers = {}
        
        for name, weight in weights.items():
            weight_shapes[name] = weight.shape
            print(f"{name}: {weight.shape}")
            
            # Collect MoE layer information
            if "mixture_of_experts" in name:
                layer_num = name.split("mixture_of_experts_")[1].split("/")[0]
                if layer_num not in moe_layers:
                    moe_layers[layer_num] = {}
                moe_layers[layer_num][name] = weight.shape
        
        # Update config based on weight shapes and MoE structure
        if moe_layers:
            print("\nDetected MoE layers:")
            for layer_num, shapes in moe_layers.items():
                print(f"Layer {layer_num}:")
                for name, shape in shapes.items():
                    print(f"  {name}: {shape}")
            
            # Update intermediate size based on MoE structure
            if any("dense" in name for name in weights.keys()):
                intermediate_layers = [shape for name, shape in weight_shapes.items() 
                                    if "dense" in name and "mixture_of_experts" in name]
                if intermediate_layers:
                    config["intermediate_size"] = max(shape[1] for shape in intermediate_layers)
                    print(f"Adjusted intermediate_size to {config['intermediate_size']}")
        
        # Find output dimension
        output_dim = None
        for name, shape in weight_shapes.items():
            if "dense" in name and len(shape) == 2:
                if shape[1] == 30522:  # This is likely the output layer
                    output_dim = shape[1]
                    break
        
        if output_dim:
            config["vocab_size"] = output_dim
            print(f"Adjusted vocab_size to {output_dim}")
            
    except Exception as e:
        print(f"Error analyzing weights: {str(e)}")
        raise
    
    # Set mixed precision policy to float32
    tf.keras.mixed_precision.set_global_policy('float32')
    
    # Create model with adjusted config
    print("\nCreating model with adjusted configuration:")
    print(f"vocab_size: {config['vocab_size']}")
    print(f"hidden_size: {config['hidden_size']}")
    print(f"intermediate_size: {config['intermediate_size']}")
    
    model = DakitariInstructModel(
        vocab_size=config["vocab_size"],
        maxlen=config["max_position_embeddings"],
        embed_dim=config["hidden_size"],
        num_heads=config["num_attention_heads"],
        ff_dim=config["intermediate_size"],
        num_transformer_blocks=config["num_hidden_layers"],
        dropout_rate=config.get("dropout_rate", 0.1)
    )
    
    # Build the model with dummy input
    dummy_input = {
        'input_ids': tf.zeros((1, config["max_position_embeddings"]), dtype=tf.int32),
        'attention_mask': tf.ones((1, config["max_position_embeddings"]), dtype=tf.int32)
    }
    model(dummy_input)
    
    print("\nModel layer shapes:")
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                print(f"{weight.name}: {weight.shape}")
    
    # Map weights to model layers with shape validation
    print("\nLoading weights into model...")
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                name = weight.name.replace(':', '_')
                if name in weights:
                    weight_value = weights[name]
                    if isinstance(weight_value, (np.ndarray, tf.Tensor)):
                        try:
                            if weight.shape != weight_value.shape:
                                print(f"Shape mismatch for {name}: Expected {weight.shape}, got {weight_value.shape}")
                                continue
                            weight.assign(weight_value)
                            print(f"Successfully loaded weights for {name}")
                        except Exception as e:
                            print(f"Error loading weights for {name}: {str(e)}")
                    else:
                        print(f"Skipping {name}: unexpected type {type(weight_value)}")
                else:
                    print(f"Warning: Weight {name} not found in safetensors file")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer, config

def convert_to_safetensors(model, output_dir):
    """Save the fine-tuned model in HuggingFace format with safetensors"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert model weights to safetensors format
    weights_dict = {}
    for weight in model.weights:
        try:
            # Get weight name and value
            name = weight.name.split(':')[0]  # Remove ':0' suffix
            
            # Convert to numpy array safely
            if isinstance(weight, tf.Variable) or isinstance(weight, tf.Tensor):
                tensor = weight.numpy()
            else:
                tensor = weight  # Already a numpy array
                
            # Map to expected HuggingFace format
            if 'embeddings' in name:
                final_key = 'embeddings'
            elif 'kernel' in name:
                final_key = 'kernel'
            elif 'bias' in name:
                final_key = 'bias'
            elif 'gamma' in name:
                final_key = 'gamma'
            elif 'beta' in name:
                final_key = 'beta'
            else:
                final_key = name
                
            weights_dict[final_key] = tensor
            print(f"Processed weight {final_key}")
            
        except Exception as e:
            print(f"Error processing weight {name}: {str(e)}")
            continue
    
    try:
        # Save weights
        safetensors_path = os.path.join(output_dir, "model.safetensors")
        save_file(weights_dict, safetensors_path)
        print(f"Saved weights to {safetensors_path}")
        
        # Save weight index
        weight_index = {
            "metadata": {"format": "tf"},
            "weight_map": {name: "model.safetensors" for name in weights_dict.keys()}
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), 'w') as f:
            json.dump(weight_index, f, indent=2)
        
        # Save model config
        config = {
            'architectures': ['DakitariInstructModel'],
            'model_type': 'dakitari_instruct',
            'vocab_size': 50000,  # Increased vocabulary
            'hidden_size': 1024,  # Larger hidden size
            'num_hidden_layers': 24,  # More layers
            'num_attention_heads': 16,  # More attention heads
            'intermediate_size': 4096,  # Larger intermediate size
            'max_position_embeddings': 1024,  # Longer sequences
            'dropout_rate': 0.1,
            'num_experts': 8,  # Added MoE capability
            'expert_capacity': 128  # Added expert capacity
        }
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Copy additional files
        src_dir = "checkpoints/Dakitari-instruct-v1.0"
        for file in ['LICENSE', 'README.md', 'configuration_dakitari_instruct.py', 
                    'modeling_dakitari_instruct.py', 'special_tokens_map.json', 
                    'vocab.txt', 'tokenizer.json', 'tokenizer_config.json']:
            src_file = os.path.join(src_dir, file)
            if os.path.exists(src_file):
                dst_file = os.path.join(output_dir, file)
                shutil.copy2(src_file, dst_file)
        
        return True
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--ff_dim", type=int, default=4096)
    parser.add_argument("--num_transformer_blocks", type=int, default=24)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--expert_capacity", type=int, default=128)
    args = parser.parse_args()

    # Load model, tokenizer and config
    model, tokenizer, config = load_model(args.model_dir)
    
    # Initialize processor with tokenizer
    processor = MedicalConversationProcessor(tokenizer=tokenizer)
    
    # Load medical dataset using correct method name
    print("Loading medical QA dataset...")
    medical_data = processor.load_medical_qa()
    
    # Prepare dataset with the loaded data
    print("Preparing training data...")
    train_data = processor.prepare_training_data(df=medical_data)
    
    # Convert the prepared data into the correct format for TensorFlow
    print("Converting to TensorFlow dataset...")
    
    # Debug information
    print("\nFirst training item structure:")
    first_item = next(iter(train_data))
    print(f"Keys: {first_item.keys() if isinstance(first_item, dict) else 'Not a dict'}")
    if isinstance(first_item, dict):
        for key, value in first_item.items():
            print(f"{key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'no shape'}")
    
    # Process data ensuring sequences
    processed_data = []
    max_length = config["max_position_embeddings"]
    
    for item in train_data:
        if not isinstance(item, dict):
            print(f"Skipping invalid item: {type(item)}")
            continue
            
        try:
            # Convert TensorFlow tensors to numpy arrays and pad
            input_ids = item['input_ids'].numpy() if isinstance(item['input_ids'], tf.Tensor) else item['input_ids']
            attention_mask = item['attention_mask'].numpy() if isinstance(item['attention_mask'], tf.Tensor) else item['attention_mask']
            labels = item['labels'].numpy() if isinstance(item['labels'], tf.Tensor) else item['labels']
            
            # Pad sequences
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            else:
                pad_length = max_length - len(input_ids)
                input_ids = np.pad(input_ids, (0, pad_length), 'constant', constant_values=0)
                attention_mask = np.pad(attention_mask, (0, pad_length), 'constant', constant_values=0)
                labels = np.pad(labels, (0, pad_length), 'constant', constant_values=-100)
            
            processed_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            continue
    
    if not processed_data:
        raise ValueError("No valid training data after processing")
    
    print(f"\nProcessed {len(processed_data)} training items")
    
    # Convert processed data to numpy arrays
    input_ids = np.array([item['input_ids'] for item in processed_data])
    attention_mask = np.array([item['attention_mask'] for item in processed_data])
    labels = np.array([item['labels'] for item in processed_data])
    
    # Create dataset dictionary
    train_dataset = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    
    # Create trainer
    trainer = AdvancedTrainer(model, tokenizer, config)
    
    # Run advanced training
    print("Starting supervised training...")
    history = trainer.supervised_training(
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate
    )
    
    # After training, convert to safetensors format
    print("\nConverting model to safetensors format...")
    if convert_to_safetensors(model, args.output_dir):
        print(f"Successfully saved model to {args.output_dir}")
    else:
        print("Failed to save model in safetensors format")

if __name__ == "__main__":
    main()
