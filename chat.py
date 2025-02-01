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

from model.transformer_model import DakitariInstructModel
from transformers import AutoTokenizer
import safetensors as sf
import json
import numpy as np

def load_model(model_dir):
    """Load the fine-tuned model and tokenizer"""
    print("Loading model...")
    
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = DakitariInstructModel(
        vocab_size=config["vocab_size"],
        maxlen=config["max_position_embeddings"],
        embed_dim=config["hidden_size"],
        num_heads=config["num_attention_heads"],
        ff_dim=config["intermediate_size"],
        num_transformer_blocks=config["num_hidden_layers"],
        dropout_rate=config.get("dropout_rate", 0.1)
    )
    
    # Load weights from safetensors
    weights_path = os.path.join(model_dir, "model.safetensors")
    weights = sf.safe_open(weights_path, framework='tensorflow')  # Specify the framework here
    
    # Map weights to model layers
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                name = weight.name.split(':')[0]  # Remove ':0' suffix
                
                # Find matching weight in safetensors
                weight_key = None
                if 'embeddings' in name:
                    weight_key = 'embeddings'
                elif 'kernel' in name:
                    weight_key = 'kernel'
                elif 'bias' in name:
                    weight_key = 'bias'
                elif 'gamma' in name:
                    weight_key = 'gamma'
                elif 'beta' in name:
                    weight_key = 'beta'
                
                if weight_key and weight_key in weights:
                    try:
                        weight.assign(weights[weight_key])
                    except Exception as e:
                        print(f"Error assigning weight {name}: {str(e)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    # Tokenize the input prompt
    inputs = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    
    # Get input_ids and attention_mask and ensure they're int32
    input_ids = tf.cast(inputs['input_ids'], tf.int32)
    attention_mask = tf.cast(inputs['attention_mask'], tf.int32)
    
    # Ensure input shapes are correct
    input_ids = tf.reshape(input_ids, (1, -1))  # Shape: (1, sequence_length)
    attention_mask = tf.reshape(attention_mask, (1, -1))  # Shape: (1, sequence_length)
    
    # Initialize generated sequence with input
    generated = input_ids
    
    # Generate tokens one by one
    for _ in range(max_length - input_ids.shape[1]):
        # Prepare model inputs
        model_inputs = {
            'input_ids': generated,
            'attention_mask': tf.ones((1, generated.shape[1]), dtype=tf.int32)
        }
        
        try:
            # Get predictions
            outputs = model(model_inputs, training=False)
            
            # Get the next token logits
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Sample from the distribution
            next_token_probs = tf.nn.softmax(next_token_logits, axis=-1)
            next_token = tf.cast(
                tf.random.categorical(next_token_logits, num_samples=1)[:, 0],
                tf.int32  # Ensure int32 type
            )
            
            # Append the new token
            next_token = tf.reshape(next_token, (1, 1))  # Reshape to match expected dimensions
            generated = tf.concat([generated, next_token], axis=1)
            
            # Stop if we predict the end token
            if next_token[0, 0] == tokenizer.eos_token_id:
                break
                
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            break
    
    # Decode the generated tokens
    try:
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error decoding response: {str(e)}")
        return prompt + " [Error: Could not generate response]"
    
def main():
    parser = argparse.ArgumentParser(description="Chat with Dakitari-Instruct Model")
    parser.add_argument("--model_dir", type=str, default="checkpoints/Dakitari-instruct-v1.1",
                       help="Path to model directory")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (higher = more random)")
    args = parser.parse_args()
    
    print("Loading model...")
    model, tokenizer = load_model(args.model_dir)
    print("Model loaded successfully!")
    
    print("\nWelcome to Dakitari-Instruct Chat!")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("Type 'clear' to clear the conversation history.")
    print("\nEnter your medical question:")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'clear':
                print("\nConversation cleared.")
                continue
            
            if not user_input:
                continue
            
            print("\nThinking...")
            response = generate_response(model, tokenizer, user_input, temperature=args.temperature)
            print(f"\nDakitari: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue

if __name__ == "__main__":
    main()
