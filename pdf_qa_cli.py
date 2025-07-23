import os
import sys
import tempfile
from pathlib import Path
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

# Constants for SmolDocling
MODEL_ID = "ds4sd/SmolDocling-256M-preview"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_pdf_pages(pdf_path):
    """Extract each page of the PDF as a PIL image."""
    images = convert_from_path(pdf_path)
    return images


def visualize_images(images):
    """Display all images using matplotlib."""
    for idx, img in enumerate(images):
        plt.figure(figsize=(8, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Page {idx + 1}')
        plt.show()


def smoldocling_analyze(image, question="Convert this page to docling."):
    """Run SmolDocling on a single image with a question/instruction."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        _attn_implementation="eager",
    ).to(DEVICE)

    # Prepare the prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=False)[0].lstrip()

    # Convert to Docling document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
    return doc


def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python pdf_qa_cli.py <pdf_path>")
    #     sys.exit(1)
    pdf_path = "analytics_data/digest_Apr_2025_EHAB CENTER - 835012.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"Extracting pages from {pdf_path}...")
    images = extract_pdf_pages(pdf_path)
    print(f"Extracted {len(images)} pages.")

    while True:
        print("\nOptions:")
        print("1. Visualize pages")
        print("2. Ask a question about a page")
        print("3. Exit")
        choice = input("Select an option (1/2/3): ").strip()
        if choice == "1":
            visualize_images(images)
        elif choice == "2":
            page_num = input(f"Enter page number (1-{len(images)}): ").strip()
            if not page_num.isdigit() or not (1 <= int(page_num) <= len(images)):
                print("Invalid page number.")
                continue
            page_idx = int(page_num) - 1
            question = input("Enter your question/instruction (or press Enter for default): ").strip()
            if not question:
                question = "Convert this page to docling."
            print(f"\nAnalyzing page {page_num} with SmolDocling...")
            doc = smoldocling_analyze(images[page_idx], question)
            print("\n--- Answer (Markdown) ---\n")
            print(doc.export_to_markdown())
            print("\n------------------------\n")
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main() 