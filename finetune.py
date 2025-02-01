import os
# Suppress TensorFlow and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Disable TensorFlow logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
import tensorflow as tf
# Disable TensorFlow warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from transformers import AutoTokenizer
from model.transformer_model import DakitariInstructModel
from data.medical_conversation import MedicalConversationProcessor
import json
from safetensors import tensorflow as sf
import numpy as np
import shutil
import glob
from tensorflow.keras import mixed_precision

# Set the policy to mixed precision
mixed_precision.set_global_policy('mixed_float16')

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Dakitari-Instruct Model")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to pre-trained model directory")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save fine-tuned model")
    return parser.parse_args()

def load_model(model_dir):
    """Load the pre-trained model and tokenizer"""
    # Load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    # Create model with parameters from config
    model = DakitariInstructModel(
        vocab_size=config["vocab_size"],  # Use vocab size from config
        maxlen=config["max_position_embeddings"],
        embed_dim=config["hidden_size"],  # Use hidden size from config
        num_heads=config["num_attention_heads"],  # Use number of attention heads from config
        ff_dim=config["intermediate_size"],  # Use intermediate size from config
        num_transformer_blocks=config["num_hidden_layers"],  # Use number of hidden layers from config
        dropout_rate=config["dropout_rate"],  # Use dropout rate from config
        num_experts=8,    # Keep MoE capability
        expert_capacity=128  # Keep expert capacity
    )
    
    # Build the model with dummy input
    dummy_input = {
        'input_ids': tf.zeros((1, config["max_position_embeddings"]), dtype=tf.int32),
        'attention_mask': tf.ones((1, config["max_position_embeddings"]), dtype=tf.int32)
    }
    model(dummy_input)
    
    # Load weights from safetensors file
    weights_path = os.path.join(model_dir, "model.safetensors")
    with open(weights_path, 'rb') as f:
        weights = sf.load_file(weights_path)
    
    # Debug info
    print("\nModel layer shapes:")
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                print(f"{weight.name}: {weight.shape}")
    
    print("\nLoaded weight shapes:")
    for name, weight in weights.items():
        print(f"{name}: {weight.shape}")
    
    # Map weights to model layers
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                name = weight.name.replace(':', '_')
                if name in weights:
                    weight_value = weights[name]
                    if isinstance(weight_value, tf.Tensor):
                        weight_value = weight_value.numpy()
                    if weight.shape == weight_value.shape:
                        try:
                            weight.assign(weight_value)
                            print(f"Successfully loaded weights for {name}")
                        except Exception as e:
                            print(f"Error loading weights for {name}: {str(e)}")
                    else:
                        print(f"\nShape mismatch for {name}:")
                        print(f"Model shape: {weight.shape}")
                        print(f"Weight shape: {weight_value.shape}")
    
    return model

def save_model(model, tokenizer, output_dir):
    """Save the fine-tuned model in HuggingFace format with safetensors"""
    import tensorflow as tf
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Convert model weights to safetensors format
    weights_dict = {}
    for weight in model.weights:
        # Get weight name and value
        name = weight.name.split(':')[0]  # Remove ':0' suffix
        tensor = weight.numpy() if isinstance(weight, tf.Variable) else weight
        
        # Map to expected HuggingFace names
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
    
    # Save weights using safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    sf.save_file(weights_dict, safetensors_path)
    
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
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'layer_norm_eps': 1e-12,
        'dropout_rate': 0.1,
        'pad_token_id': 0,
        'bos_token_id': 1,
        'eos_token_id': 2
    }
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save generation config
    generation_config = {
        "_from_model_config": True,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2
    }
    gen_config_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_config_path, 'w') as f:
        json.dump(generation_config, f, indent=2)
    
    # Copy additional files from original checkpoint
    src_dir = "checkpoints/Dakitari-instruct-v1.0"
    for file in ['LICENSE', 'README.md', 'configuration_dakitari_instruct.py', 
                'modeling_dakitari_instruct.py', 'special_tokens_map.json', 
                'vocab.txt', 'tokenizer.json', 'tokenizer_config.json']:
        src_file = os.path.join(src_dir, file)
        if os.path.exists(src_file):
            dst_file = os.path.join(output_dir, file)
            shutil.copy2(src_file, dst_file)
    
    print(f"Model saved in HuggingFace format to {output_dir}")

def main():
    args = parse_args()
    
    # Load tokenizer and ensure vocab size matches model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    vocab_size = 512  # Model's vocabulary size
    
    # Load pre-trained model
    model = load_model(args.model_dir)
    
    # Prepare medical conversation dataset
    processor = MedicalConversationProcessor(tokenizer, args.max_length)
    medical_data = processor.load_medical_qa()
    
    # Prepare training data
    texts = medical_data['text'].tolist()
    
    # Tokenize all texts with truncation
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=args.max_length,
        return_tensors="tf"
    )
    
    # Ensure token indices are within vocabulary bounds
    input_ids = tf.clip_by_value(
        tf.cast(encodings['input_ids'], tf.int32),
        clip_value_min=0,
        clip_value_max=vocab_size - 1
    )
    attention_mask = tf.cast(encodings['attention_mask'], tf.int32)
    labels = tf.clip_by_value(
        tf.cast(input_ids, tf.int32),
        clip_value_min=0,
        clip_value_max=vocab_size - 1
    )
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        },
        labels
    ))
    
    # Prepare training
    train_dataset = dataset.shuffle(1000).batch(args.batch_size)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    def masked_loss(y_true, y_pred):
        y_true = tf.clip_by_value(
            tf.cast(y_true, tf.int32),
            clip_value_min=0,
            clip_value_max=vocab_size - 1
        )
        mask = tf.cast(y_true != 0, tf.float16)  # Ensure mask is float16
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    
    def masked_accuracy(y_true, y_pred):
        y_true = tf.clip_by_value(
            tf.cast(y_true, tf.int32),
            clip_value_min=0,
            clip_value_max=vocab_size - 1
        )
        mask = tf.cast(y_true != 0, tf.float32)
        predictions = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        correct = tf.cast(predictions == y_true, tf.float32) * mask
        return tf.reduce_sum(correct) / tf.reduce_sum(mask)
    
    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    
    # Train
    print("Starting fine-tuning...")
    print(f"Training on {len(texts)} examples")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    
    model.fit(
        train_dataset,
        epochs=args.epochs
    )
    
    # Save fine-tuned model
    save_model(model, tokenizer, args.output_dir)
    print(f"Fine-tuned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
