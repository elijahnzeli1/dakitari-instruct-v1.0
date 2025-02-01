---
language:
- en
license:
- mit
task_categories:
- text-generation
- question-answering
task_ids:
- language-modeling
- question-answering
size_categories:
- 10M<n<100M
tags:
- medical
- healthcare
- pubmed
- biomedical
pretty_name: Dakitari-Instruct Medical Dataset
---

# Dakitari-Instruct Medical Dataset

## Dataset Description

- **Repository:** Dakitari-Instruct Medical Dataset
- **Languages:** English
- **Tasks:** Medical Text Generation, Question Answering
- **Size:** >10M samples
- **Version:** 1.0.0
- **Last Updated:** 2024
- **License:** MIT

## Dataset Summary

This dataset is a combination of medical literature from PubMed abstracts and medical question-answering pairs, specifically curated for training the Dakitari-Instruct model. The dataset focuses on medical and biomedical content to enable high-quality medical text generation and question answering capabilities.

## Dataset Sources

The dataset combines two main sources:

1. **PubMed Abstracts:**
   - Source: PubMed/MEDLINE database
   - Content: Scientific abstracts from biomedical literature
   - Type: Research papers, clinical studies, medical reviews

2. **Medical QA Pairs:**
   - Source: MedQuAD and PubMedQA datasets
   - Content: Medical questions and their corresponding answers
   - Type: Clinical questions, patient queries, medical explanations

## Dataset Structure

The dataset is structured as follows:

```python
{
    'text': str,  # Combined text from abstracts and QA pairs
    'input_ids': tensor,  # Tokenized input sequences
    'attention_mask': tensor  # Attention masks for the sequences
}
```

## Data Fields

- `text`: String containing either a PubMed abstract or a question-answer pair
- `input_ids`: Tokenized representation of the text
- `attention_mask`: Mask indicating which tokens should be attended to

## Data Splits

- Training: 90% of the data
- Validation: 10% of the data

## Dataset Creation

### Preprocessing

1. Text Extraction:
   - PubMed abstracts are extracted from the source datasets
   - QA pairs are combined into single text sequences
   - All texts are tokenized using the BiomedNLP-PubMedBERT tokenizer

2. Data Cleaning:
   - Removal of duplicate entries
   - Handling of missing values
   - Length normalization

### Quality Control

- Validation of text lengths
- Verification of tokenization quality
- Checking for data consistency

## Considerations for Use

### Medical Context

This dataset contains medical information and should be used with appropriate consideration for:
- Medical accuracy
- Patient privacy
- Healthcare context
- Professional medical guidance

### Limitations

- The dataset may contain technical medical terminology
- Some medical concepts may be outdated
- The dataset is primarily in English
- Coverage may vary across medical specialties

## Additional Information

### Dataset Curators

The dataset was curated by the Dakitari-Instruct team, combining publicly available medical datasets from PubMed and medical QA sources.

### Licensing Information

This dataset is released under the MIT License. However, please note that individual source datasets may have their own licensing terms:
- PubMed data usage terms: https://www.nlm.nih.gov/databases/download/terms_and_conditions.html
- Individual dataset licenses should be consulted for specific use cases

### Citation Information

If you use this dataset, please cite:

```bibtex
@misc{dakitari2024medical,
  title={Dakitari-Instruct Medical Dataset},
  year={2024},
  publisher={Hugging Face},
  note={A combined dataset of PubMed abstracts and medical QA pairs}
}
```

### Contributions

Thanks to:
- The PubMed/MEDLINE database for medical abstracts
- The medical community for QA datasets
- The Hugging Face team for infrastructure support 