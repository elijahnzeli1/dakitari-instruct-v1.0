---
annotations_creators:
- machine-generated
language:
- en
language_creators:
- machine-generated
license:
- mit
multilinguality:
- monolingual
pretty_name: Dakitari-Instruct
size_categories:
- 10M<n<100M
source_datasets:
- hwang2006/PUBMED_title_abstracts_2020_baseline
- vaishnavm/medquad
- pubmed_qa
task_categories:
- text-generation
- question-answering
task_ids:
- language-modeling
- medical-qa
---

# Dakitari-Instruct Medical Language Model

A specialized language model for medical text generation and question answering, trained on PubMed abstracts and medical QA datasets.

## Model Description

- **Model Type:** Transformer-based Language Model
- **Language:** English
- **License:** MIT
- **Training Data:** PubMed abstracts + Medical QA pairs

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModelForCausalLM.from_pretrained("path/to/dakitari-instruct")
```

## Training Data

This model is trained on:
1. PubMed abstracts (biomedical literature)
2. Medical question-answer pairs

For detailed dataset information, see [dataset_card.md](dataset_card.md).

## Features

- Medical text generation
- Medical question answering
- Research assistance
- Built on transformer architecture
- Trained on PubMed abstracts and medical Q&A datasets

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Transformers library
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/elijahnzeli1/dakitari-instruct.git
cd dakitari-instruct
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
#for windows
.\venv\Scripts\Activate
#for mac/linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

1. Start training:
```bash
python train.py --batch_size 32 --epochs 10
```

2. Monitor training progress with TensorBoard:
```bash
tensorboard --logdir logs
```
Then open your browser and navigate to `http://localhost:6006` to view training metrics.

Training logs are also saved in CSV format in the `logs` directory for backup.

### Training Parameters

You can customize the training by adjusting these parameters:

- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--max_length`: Maximum sequence length (default: 512)
- `--embed_dim`: Embedding dimension (default: 256)
- `--num_heads`: Number of attention heads (default: 8)
- `--ff_dim`: Feed-forward dimension (default: 512)
- `--num_transformer_blocks`: Number of transformer blocks (default: 6)
- `--dropout_rate`: Dropout rate (default: 0.1)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--checkpoint_dir`: Directory to save model checkpoints (default: "checkpoints")
- `--log_dir`: Directory to save training logs (default: "logs")

## Project Structure

```
dakitari-instruct/
├── data/
│   └── preprocess.py      # Data preprocessing utilities
├── model/
│   └── transformer_model.py # Model architecture
├── train.py              # Training script
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Hardware Requirements

- Minimum: 16GB RAM, NVIDIA GPU with 8GB VRAM
- Recommended: 32GB RAM, NVIDIA GPU with 16GB+ VRAM

## Dataset Sources

The model is trained on:
1. PubMed abstracts
2. Medical questions and answers pairs

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the HuggingFace team for the transformers library
- Thanks to the TensorFlow team for the excellent framework
- Thanks to the medical community for the valuable datasets 

Now you can track your training progress locally using TensorBoard, which provides:
- Real-time metrics visualization
- Learning rate tracking
- Model graph visualization
- Histogram of weights and biases

The training metrics are also saved in CSV format as a backup, which you can analyze using any spreadsheet software or data analysis tools.
To view the training progress:
- Start training your model
- In a separate terminal, run [tensorboard --logdir logs]()
- Open <http://localhost:6006> in your browser


## DO FINETUNNING OF MODEL REDUCE THE SIZE OF A MODEL.SAFETENSORS 

Fine-tuning a model can potentially reduce the size of the saved model file (like model.safetensors) depending on several factors:

Pruning: If you implement techniques such as pruning during fine-tuning, you can remove less significant weights, which can reduce the overall size of the model.
Quantization: Fine-tuning can also involve quantization, where weights are stored in lower precision (e.g., from float32 to int8). This can significantly reduce the model size while maintaining acceptable performance.
Weight Sharing: Some fine-tuning approaches may involve weight sharing, which can also lead to a smaller model size.
Training Data: If the fine-tuning process leads to a more compact representation of the learned weights (i.e., fewer parameters), the resulting model file may be smaller.
