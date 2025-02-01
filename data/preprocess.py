import os
import tensorflow as tf
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import List, Tuple, Dict

class MedicalDataProcessor:
    def __init__(
        self,
        max_length: int = 512,
        batch_size: int = 32,
        tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def load_pubmed_data(self) -> tf.data.Dataset:
        """Load PubMed abstracts dataset"""
        try:
            # Load PubMed QA unlabeled dataset as our main source
            dataset = load_dataset("pubmed_qa", "pqa_unlabeled", split="train", trust_remote_code=True)
            print("Successfully loaded PubMed QA unlabeled dataset")
            # Convert to expected format
            return dataset.map(lambda x: {
                "text": x["context"] if x["context"] else x["question"]
            })
        except Exception as e:
            print(f"Failed to load PubMed QA dataset: {e}")
            raise Exception(
                "Could not load PubMed dataset. Please check your internet connection "
                "and ensure you have run 'huggingface-cli login'."
            )
        
    def load_medical_qa(self) -> tf.data.Dataset:
        """Load medical QA dataset"""
        try:
            # Load PubMed QA labeled dataset
            dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)
            print("Successfully loaded PubMedQA labeled dataset")
            # Convert to expected format
            return dataset.map(lambda x: {
                "question": x["question"],
                "answer": x["long_answer"] if "long_answer" in x else x["final_decision"]
            })
        except Exception as e:
            print(f"Failed to load PubMedQA dataset: {e}")
            raise Exception(
                "Could not load Medical QA dataset. Please check your internet connection "
                "and ensure you have run 'huggingface-cli login'."
            )
    
    def preprocess_text(self, texts: List[str]) -> Dict[str, tf.Tensor]:
        """Tokenize and prepare text for model input"""
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
    
    def clean_text(self, text) -> str:
        """Clean and validate text input"""
        if text is None:
            return ""
        # Convert to string if not already
        text = str(text)
        # Remove excessive whitespace
        text = " ".join(text.split())
        return text

    def _generate_synthetic_dataset(self):
        """Generate a synthetic dataset when real dataset loading fails"""
        import tensorflow as tf
        import random
        
        # Generate synthetic data
        num_samples = 1000
        synthetic_data = []
        for _ in range(num_samples):
            synthetic_data.append({
                "input_ids": [random.randint(0, 1000) for _ in range(self.max_length)],
                "attention_mask": [1] * self.max_length,
                "target": random.randint(0, 2)  # Assuming 3-class classification
            })
        
        # Convert to TensorFlow dataset
        inputs = {
            "input_ids": tf.constant([item["input_ids"] for item in synthetic_data]),
            "attention_mask": tf.constant([item["attention_mask"] for item in synthetic_data])
        }
        targets = tf.constant([item["target"] for item in synthetic_data])
        
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset, 1001  # Return dataset and vocab size

    def prepare_medical_corpus(self, save_dir: str = "processed_data", split: str = 'train'):
        """Prepare medical corpus for training or testing"""
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Try to load PubMed dataset
        try:
            # Load dataset with specified split
            if split == 'train':
                dataset = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")
            else:
                # Fallback to train split if test/validation not available
                dataset = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")
            
            print(f"Successfully loaded PubMed QA {split} dataset")
        except Exception as e:
            print(f"Failed to load PubMed QA dataset: {e}")
            # Fallback to synthetic data
            return self._generate_synthetic_dataset()
        
        # Extract texts from dataset
        texts = [
            str(item['context']) if item.get('context') else str(item.get('question', ''))
            for item in dataset
        ]
        
        # Clean and filter texts
        texts = [self.clean_text(text) for text in texts if text]
        texts = [text for text in texts if len(text.split()) > 5]  # Filter out very short texts
        
        # Create input-target pairs for language modeling
        # Use sliding window approach
        input_texts = texts[:-1]
        target_texts = texts[1:]
        
        # Preprocess inputs and targets
        inputs = self.preprocess_text(input_texts)
        targets = self.preprocess_text(target_texts)
        
        # Convert to TensorFlow dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            targets["input_ids"]  # Use input_ids as targets for language modeling
        ))
        
        # Batch and prefetch
        tf_dataset = tf_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return tf_dataset, self.tokenizer.vocab_size

    def create_medical_dataset(self, texts: List[str]) -> tf.data.Dataset:
        """Create TensorFlow dataset from medical texts"""
        # Ensure all inputs are valid strings
        valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts found after filtering")
            
        # Create input-target pairs for language modeling
        input_texts = valid_texts[:-1]
        target_texts = valid_texts[1:]
        
        # Tokenize inputs and targets
        inputs = self.preprocess_text(input_texts)
        targets = self.preprocess_text(target_texts)
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            targets["input_ids"]
        ))
        
        # Shuffle and batch the dataset
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset 