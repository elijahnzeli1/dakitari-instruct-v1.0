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
    except RuntimeError as e:
        print(f"Memory growth setting failed: {e}")

import numpy as np
import datetime
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.transformer_model import DakitariInstructModel
from data.preprocess import MedicalDataProcessor
import argparse
import safetensors.tensorflow

def parse_args():
    parser = argparse.ArgumentParser(description="Train Dakitari-Instruct Model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--num_transformer_blocks", type=int, default=6)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    return parser.parse_args()

def create_model(vocab_size, args):
    model = DakitariInstructModel(
        vocab_size=vocab_size,
        maxlen=args.max_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_transformer_blocks=args.num_transformer_blocks,
        dropout_rate=args.dropout_rate
    )
    return model

class SafetensorsModelCheckpoint(tf.keras.callbacks.Callback):
    """Custom callback to save model weights in safetensors format"""
    def __init__(self, checkpoint_dir, save_freq='epoch', save_best_only=True, monitor='val_loss'):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best = float('inf')
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        from safetensors.tensorflow import save_file
        import tensorflow as tf
        
        # Only save if we've improved (if save_best_only is True)
        current = logs.get(self.monitor)
        if self.save_best_only and current >= self.best:
            return
        
        self.best = current
        
        # Create weights dictionary
        weights_dict = {}
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                for weight in layer.weights:
                    name = weight.name.replace(':', '_')
                    # Convert to numpy array safely
                    weight_value = weight
                    if isinstance(weight_value, tf.Tensor):
                        weight_value = weight_value.numpy()
                    weights_dict[name] = weight_value
        
        # Save weights
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"model_epoch_{epoch:02d}_loss_{current:.4f}.safetensors"
        )
        save_file(weights_dict, checkpoint_path)
        
        # Save weight map index
        weight_map = {name: os.path.basename(checkpoint_path) for name in weights_dict.keys()}
        index = {
            "metadata": {"format": "tf", "epoch": epoch, "loss": float(current)},
            "weight_map": weight_map
        }
        
        index_path = os.path.join(
            self.checkpoint_dir,
            f"model_epoch_{epoch:02d}_loss_{current:.4f}.safetensors.index.json"
        )
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        
        print(f"\nSaved checkpoint for epoch {epoch} to {checkpoint_path}")

def train_model(model, dataset, args):
    """Train the model with the given dataset"""
    # Create log directory with timestamp
    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Create TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Create CSV logger for backup
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(log_dir, 'training_history.csv'),
        separator=',',
        append=False
    )
    
    # Create safetensors checkpoint callback
    checkpoint_callback = SafetensorsModelCheckpoint(
        checkpoint_dir=args.checkpoint_dir,
        save_freq='epoch',
        save_best_only=True,
        monitor='val_loss'
    )
    
    # Compile model with sparse categorical crossentropy loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    # Calculate dataset size and split sizes
    dataset_size = sum(1 for _ in dataset)
    val_size = int(dataset_size * 0.1)
    train_size = dataset_size - val_size
    
    # Split dataset into train and validation
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    
    # Train model with input-target pairs
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        callbacks=[checkpoint_callback, tensorboard_callback, csv_logger],
        validation_data=val_dataset,
        verbose=1
    )
    
    return history

def save_model(model, tokenizer, args):
    """Save model and all required files in Hugging Face format"""
    import tensorflow as tf
    from safetensors.tensorflow import save_file
    
    # Create model save directory
    model_save_path = os.path.join(args.checkpoint_dir, "Dakitari-instruct-v1.0")
    os.makedirs(model_save_path, exist_ok=True)
    
    # Create weights dictionary
    weights_dict = {}
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                name = weight.name.replace(':', '_')
                # Handle both TensorFlow tensors and numpy arrays
                if isinstance(weight, tf.Tensor):
                    weight_value = weight.numpy()
                else:
                    weight_value = weight
                weights_dict[name] = weight_value
    
    # Save weights using safetensors
    safetensors_path = os.path.join(model_save_path, "model.safetensors")
    save_file(weights_dict, safetensors_path)
    
    # Save weight map index
    weight_map = {name: "model.safetensors" for name in weights_dict.keys()}
    index = {
        "metadata": {"format": "tf"},
        "weight_map": weight_map
    }
    
    with open(os.path.join(model_save_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    
    # Save LICENSE
    license_content = """MIT License

Copyright (c) 2025 Dakitari-Instruct

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files..."""
    
    with open(os.path.join(model_save_path, "LICENSE"), "w") as f:
        f.write(license_content)
    
    # Save README.md
    readme_content = """# Dakitari-Instruct v1.0

A medical domain-specific language model trained on PubMed data..."""
    
    with open(os.path.join(model_save_path, "README.md"), "w") as f:
        f.write(readme_content)
    
    # Save config.json
    config = {
        "architectures": ["DakitariInstructModel"],
        "model_type": "dakitari_instruct",
        "vocab_size": model.vocab_size,
        "hidden_size": model.embed_dim,
        "num_hidden_layers": model.num_transformer_blocks,
        "num_attention_heads": model.num_heads,
        "intermediate_size": model.ff_dim,
        "max_position_embeddings": model.maxlen,
        "layer_norm_eps": 1e-6,
        "dropout": model.dropout_rate,
        "initializer_range": 0.02,
        "use_cache": True
    }
    
    with open(os.path.join(model_save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Save configuration_dakitari_instruct.py
    config_py_content = """from transformers import PretrainedConfig

class DakitariInstructConfig(PretrainedConfig):
    model_type = "dakitari_instruct"
    
    def __init__(self, vocab_size=30522, hidden_size=256, num_hidden_layers=6,
                 num_attention_heads=8, intermediate_size=512, max_position_embeddings=512,
                 layer_norm_eps=1e-6, dropout=0.1, initializer_range=0.02, use_cache=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache"""
    
    with open(os.path.join(model_save_path, "configuration_dakitari_instruct.py"), "w") as f:
        f.write(config_py_content)
    
    # Save generation_config.json
    generation_config = {
        "max_length": model.maxlen,
        "num_beams": 4,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3
    }
    
    with open(os.path.join(model_save_path, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)
    
    # Save modeling_dakitari_instruct.py
    modeling_py_content = """from transformers import PreTrainedModel
from .configuration_dakitari_instruct import DakitariInstructConfig

class DakitariInstructPreTrainedModel(PreTrainedModel):
    config_class = DakitariInstructConfig
    base_model_prefix = "dakitari_instruct"
    
    def __init__(self, config):
        super().__init__(config)"""
    
    with open(os.path.join(model_save_path, "modeling_dakitari_instruct.py"), "w") as f:
        f.write(modeling_py_content)
    
    # Save tokenizer
    tokenizer.save_pretrained(model_save_path)
    
    print(f"Model saved to {model_save_path}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize data processor
    data_processor = MedicalDataProcessor(
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    # Prepare dataset
    dataset, vocab_size = data_processor.prepare_medical_corpus()
    
    # Create and train model
    model = create_model(vocab_size, args)
    history = train_model(model, dataset, args)
    
    # Save model using Hugging Face method
    save_model(model, data_processor.tokenizer, args)

if __name__ == "__main__":
    main()