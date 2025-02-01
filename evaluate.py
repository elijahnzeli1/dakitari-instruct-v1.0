import os
# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable AVX2 and FMA instructions
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Configure TensorFlow to use available CPU instructions
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': True,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
    'arithmetic_optimization': True,
    'dependency_optimization': True,
    'loop_optimization': True,
    'function_optimization': True,
    'debug_stripper': True
})

# Attempt to enable XLA (Accelerated Linear Algebra) compilation
tf.config.optimizer.set_jit(True)

from model.transformer_model import DakitariInstructModel
from data.preprocess import MedicalDataProcessor

def load_dakitari_model(model_path):
    """
    Load Dakitari Instruct Model with Mixture of Experts architecture
    
    Args:
        model_path (str): Path to the model directory
    """
    import json
    from safetensors.tensorflow import load_file
    from model.transformer_model import (
        DakitariInstructModel
    )
    
    # Load config
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create model with config parameters
    model = DakitariInstructModel(
        vocab_size=config["vocab_size"],
        maxlen=config["max_position_embeddings"],
        embed_dim=config["hidden_size"],
        num_heads=config["num_attention_heads"],
        ff_dim=config["intermediate_size"],
        num_transformer_blocks=config["num_hidden_layers"],
        dropout_rate=config["dropout"]
    )
    
    # Build the model
    dummy_input = {
        'input_ids': tf.zeros((1, config["max_position_embeddings"]), dtype=tf.int32),
        'attention_mask': tf.ones((1, config["max_position_embeddings"]), dtype=tf.int32)
    }
    model(dummy_input)
    
    # Load weights
    try:
        weights = load_file(model_path)
        
        # Map weights to model layers
        weight_value_tuples = []
        for layer in model.layers:
            if hasattr(layer, 'weights'):
                for weight in layer.weights:
                    name = weight.name.replace(':', '_')
                    if name in weights:
                        weight_value_tuples.append((weight, weights[name]))
        
        keras.backend.batch_set_value(weight_value_tuples)
        print(f"Successfully loaded weights from {model_path}")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        raise
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model

def evaluate_model(model, dataset, args):
    """
    Evaluate the model on the given dataset
    
    Args:
        model: DakitariInstructModel instance
        dataset: tf.data.Dataset instance
        args: Parsed command line arguments
    """
    # Prepare metrics
    loss_metric = keras.metrics.Mean()
    accuracy_metric = keras.metrics.SparseCategoricalAccuracy()
    perplexity_metric = keras.metrics.Mean()
    
    # Create progress bar
    total_batches = sum(1 for _ in dataset)
    progress_bar = tf.keras.utils.Progbar(total_batches)
    
    for batch_idx, (inputs, targets) in enumerate(dataset):
        # Forward pass
        predictions = model(inputs, training=False)
        
        # Calculate loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            targets, predictions, from_logits=False
        )
        loss = tf.reduce_mean(loss)
        
        # Calculate perplexity
        perplexity = tf.exp(loss)
        
        # Update metrics
        loss_metric.update_state(loss)
        accuracy_metric.update_state(targets, predictions)
        perplexity_metric.update_state(perplexity)
        
        # Update progress bar
        progress_bar.update(batch_idx + 1, [
            ('loss', loss_metric.result()),
            ('accuracy', accuracy_metric.result()),
            ('perplexity', perplexity_metric.result())
        ])
    
    # Print final results
    print("\nEvaluation Results:")
    print(f"Loss: {loss_metric.result():.4f}")
    print(f"Accuracy: {accuracy_metric.result():.4f}")
    print(f"Perplexity: {perplexity_metric.result():.4f}")
    
    # Generate sample predictions
    print("\nSample Predictions:")
    sample_inputs = next(iter(dataset))[0]
    sample_outputs = model.generate(
        sample_inputs,
        max_length=args.max_length,
        num_beams=4,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    
    return {
        'loss': float(loss_metric.result()),
        'accuracy': float(accuracy_metric.result()),
        'perplexity': float(perplexity_metric.result())
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dakitari-Instruct Model")
    parser.add_argument("--model_path", type=str, 
                        default="checkpoints/model.sensors", 
                        help="Path to the model checkpoint")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize data processor
    processor = MedicalDataProcessor(maxlen=args.max_length)
    
    # Load evaluation dataset
    dataset = processor.prepare_medical_corpus(split='test')
    dataset = dataset.batch(args.batch_size)
    
    print(f"Loading model from {args.model_path}")
    model = load_dakitari_model(args.model_path)
    
    print("Starting evaluation...")
    metrics = evaluate_model(model, dataset, args)
    
    # Save metrics to file
    output_dir = os.path.dirname(args.model_path)
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    
    print(f"Evaluation metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
