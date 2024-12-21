from bs4 import BeautifulSoup
import os
import requests
import json
from transformers import CLIPProcessor, CLIPTextModelWithProjection
import torch

def save_to_json(structured_content, output_file='output.json'):
    """
    Save structured content to a JSON file.
    
    Args:
        structured_content (list): List of dictionaries containing structured content
        output_file (str): Path to the output JSON file (default: 'output.json')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Save to JSON file with proper formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structured_content, f, indent=4, ensure_ascii=False)

def load_from_json(input_file):
    """
        Load structured content from a JSON file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def embed_text(text):
    """
        Convet text to embeddings using CLIP
    """
    
    # import model
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    # import processor (handles text tokenization and image preprocessing)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    # pre-process text and images
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    # compute embeddings with CLIP
    outputs = model(**inputs)

    return outputs.text_embeds