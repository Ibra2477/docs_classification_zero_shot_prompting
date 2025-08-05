#!/usr/bin/env python3
"""
Document OCR Converter using Tesseract
Converts images and documents to plain text and saves them locally.
Enhanced for better multi-page document handling.
"""

import os
import sys
from pathlib import Path
import argparse
from PIL import Image
import pytesseract
import cv2

# Configure Tesseract path for Windows
import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import numpy as np
from pdf2image import convert_from_path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentOCRConverter:
    def __init__(self, input_dir="data/images", output_dir="output", languages="eng", 
                 preserve_structure=True, page_separator="\n\n--- Page {} ---\n\n"):
        """
        Initialize the OCR converter.
        
        Args:
            input_dir (str): Directory containing input documents
            output_dir (str): Directory to save converted text files
            languages (str): Tesseract language codes (e.g., 'eng', 'eng+fra')
            preserve_structure (bool): Whether to preserve page structure for multi-page docs
            page_separator (str): Separator template for pages (use {} for page number)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.languages = languages
        self.preserve_structure = preserve_structure
        self.page_separator = page_separator
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf'}
        
        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify Tesseract installation and languages
        self._verify_tesseract()
    
    def _verify_tesseract(self):
        """Verify that Tesseract is properly installed and languages are available."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR version: {version}")
            
            # List installed languages
            langs = pytesseract.get_languages(config='')
            logger.info(f"Available languages: {', '.join(langs)}")
            logger.info(f"Using languages: {self.languages}")
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            logger.error("Please install Tesseract OCR and language packs:")
            logger.error("  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-fra")
            logger.error("  macOS: brew install tesseract tesseract-lang")
            logger.error("  Windows: Download language packs from https://github.com/tesseract-ocr/tessdata")
            sys.exit(1)
    
    def preprocess_image(self, image_input):
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image_input: Either a file path (Path/str) or numpy array
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Handle different input types
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not read image: {image_input}")
        else:
            img = image_input
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Adaptive thresholding works better for varied lighting
        processed = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return processed
    
    def extract_text_from_processed_image(self, processed_image):
        """
        Extract text from a preprocessed image.
        
        Args:
            processed_image: Preprocessed image (numpy array)
            
        Returns:
            str: Extracted text
        """
        try:
            # Use auto page segmentation
            custom_config = r'--oem 3 --psm 3'
            
            text = pytesseract.image_to_string(
                processed_image, 
                lang=self.languages, 
                config=custom_config
            )
            
            if not text.strip():
                logger.warning("Tesseract returned empty text. Check image quality and language settings.")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def extract_text_from_image(self, image_path):
        """
        Extract text from a single image file.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            str: Extracted text
        """
        try:
            processed_image = self.preprocess_image(image_path)
            
            # Debug: Save preprocessed image
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            debug_path = debug_dir / f"{image_path.stem}_preprocessed.png"
            cv2.imwrite(str(debug_path), processed_image)
            logger.info(f"Saved preprocessed image to: {debug_path}")
            
            return self.extract_text_from_processed_image(processed_image)
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return ""
    
    def detect_multi_page_tiff(self, image_path):
        """
        Check if TIFF file has multiple pages.
        
        Args:
            image_path (Path): Path to the TIFF file
            
        Returns:
            tuple: (is_multipage, page_count)
        """
        try:
            with Image.open(image_path) as img:
                page_count = 1
                try:
                    while True:
                        img.seek(page_count)
                        page_count += 1
                except EOFError:
                    pass
                return page_count > 1, page_count
        except Exception:
            return False, 1
    
    def extract_text_from_multipage_tiff(self, image_path):
        """
        Extract text from multi-page TIFF file.
        
        Args:
            image_path (Path): Path to the TIFF file
            
        Returns:
            str: Extracted text from all pages
        """
        try:
            all_text = []
            
            with Image.open(image_path) as img:
                page_num = 0
                try:
                    while True:
                        img.seek(page_num)
                        logger.info(f"Processing TIFF page {page_num + 1} of {image_path.name}")
                        
                        # Convert PIL image to OpenCV format
                        page_array = np.array(img.convert('RGB'))
                        page_cv = cv2.cvtColor(page_array, cv2.COLOR_RGB2BGR)
                        
                        processed_image = self.preprocess_image(page_cv)
                        text = self.extract_text_from_processed_image(processed_image)
                        
                        if text.strip():
                            if self.preserve_structure and len(all_text) > 0:
                                all_text.append(self.page_separator.format(page_num + 1))
                            all_text.append(text.strip())
                        
                        page_num += 1
                        
                except EOFError:
                    pass
            
            return "\n\n".join(all_text) if all_text else ""
            
        except Exception as e:
            logger.error(f"Error extracting text from multi-page TIFF {image_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF by converting to images first.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            str: Extracted text from all pages
        """
        try:
            # Convert PDF to images - only do this once
            pages = convert_from_path(
                str(pdf_path), 
                dpi=300,
                poppler_path=r'C:\Program Files\poppler-23.11.0\Library\bin' if platform.system() == "Windows" else None
            )
            logger.info(f"Converted PDF to {len(pages)} pages")
            
            if not pages:
                logger.error(f"No pages converted from PDF: {pdf_path}")
                return ""
            
            all_text = []
            for i, page in enumerate(pages):
                logger.info(f"Processing PDF page {i+1}/{len(pages)} of {pdf_path.name}")
                
                # Convert PIL image to OpenCV format
                page_array = np.array(page)
                page_cv = cv2.cvtColor(page_array, cv2.COLOR_RGB2BGR)
                
                processed_image = self.preprocess_image(page_cv)
                text = self.extract_text_from_processed_image(processed_image)
                
                if text.strip():
                    if self.preserve_structure and len(all_text) > 0:
                        all_text.append(self.page_separator.format(i + 1))
                    all_text.append(text.strip())
            
            return "\n\n".join(all_text) if all_text else ""
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def process_single_file(self, file_path):
        """
        Process a single file and extract text.
        
        Args:
            file_path (Path): Path to the file to process
            
        Returns:
            tuple: (extracted_text, is_multipage, page_count)
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            text = self.extract_text_from_pdf(file_path)
            return text, len(text) > 0, 1  # Don't double-convert PDFs
            
        elif file_extension in {'.tiff', '.tif'}:
            is_multipage, page_count = self.detect_multi_page_tiff(file_path)
            if is_multipage:
                text = self.extract_text_from_multipage_tiff(file_path)
                return text, True, page_count
            else:
                text = self.extract_text_from_image(file_path)
                return text, False, 1
                
        elif file_extension in {'.jpg', '.jpeg', '.png', '.bmp'}:
            text = self.extract_text_from_image(file_path)
            return text, False, 1
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return "", False, 0
    
    def save_extracted_text(self, file_path, extracted_text, is_multipage=False, page_count=1):
        """
        Save extracted text to output file with metadata.
        
        Args:
            file_path (Path): Original file path
            extracted_text (str): Extracted text content
            is_multipage (bool): Whether source was multi-page
            page_count (int): Number of pages processed
        """
        output_filename = f"{file_path.stem}.txt"
        output_path = self.output_dir / output_filename
        
        try:
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Source: {file_path.name}\n")
                f.write(f"Processed: {current_time}\n")
                f.write(f"Document Type: {'Multi-page' if is_multipage else 'Single-page'}\n")
                f.write(f"Page Count: {page_count}\n")
                f.write(f"Language: {self.languages}\n")
                f.write("=" * 50 + "\n\n")
                f.write(extracted_text)
            
            logger.info(f"Saved text to: {output_path} ({len(extracted_text)} characters)")
            
        except Exception as e:
            logger.error(f"Error saving text file {output_path}: {e}")
    
    def convert_documents(self):
        """
        Convert all supported documents in the input directory.
        """
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return
        
        # Find all supported files without duplicates
        seen = set()
        supported_files = []
        for ext in self.supported_formats:
            for p in self.input_dir.glob(f"*{ext}"):
                if p.name.lower() not in seen:
                    seen.add(p.name.lower())
                    supported_files.append(p)
            for p in self.input_dir.glob(f"*{ext.upper()}"):
                if p.name.lower() not in seen:
                    seen.add(p.name.lower())
                    supported_files.append(p)
        
        if not supported_files:
            logger.warning(f"No supported files found in {self.input_dir}")
            logger.info(f"Supported formats: {', '.join(self.supported_formats)}")
            return
        
        logger.info(f"Found {len(supported_files)} files to process")
        
        total_pages = 0
        processed_files = 0
        
        for file_path in supported_files:
            logger.info(f"Processing: {file_path.name}")
            
            extracted_text, is_multipage, page_count = self.process_single_file(file_path)
            
            if extracted_text:
                self.save_extracted_text(file_path, extracted_text, is_multipage, page_count)
                total_pages += page_count
                processed_files += 1
            else:
                logger.warning(f"No text extracted from {file_path.name}")
        
        logger.info(f"Processing complete! {processed_files} files processed, {total_pages} total pages")
    
    def process_single_document(self, file_path):
        """
        Process a single document specified by path.
        
        Args:
            file_path (str): Path to the document to process
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return
        
        if file_path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return
        
        logger.info(f"Processing single file: {file_path}")
        
        extracted_text, is_multipage, page_count = self.process_single_file(file_path)
        
        if extracted_text:
            self.save_extracted_text(file_path, extracted_text, is_multipage, page_count)
        else:
            logger.warning(f"No text extracted from {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert documents to plain text using OCR")
    parser.add_argument("--input-dir", default="data/images", help="Input directory containing documents")
    parser.add_argument("--output-dir", default="output", help="Output directory for text files")
    parser.add_argument("--languages", default="eng+fra", help="Tesseract language codes (e.g., 'eng', 'eng+fra')")
    parser.add_argument("--file", help="Process a single file instead of a directory")
    parser.add_argument("--no-structure", action="store_true", help="Don't preserve page structure (merge all text)")
    parser.add_argument("--page-separator", default="\n\n--- Page {} ---\n\n", 
                       help="Custom page separator (use {} for page number)")
    
    args = parser.parse_args()
    
    # Create converter instance
    converter = DocumentOCRConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        languages=args.languages,
        preserve_structure=not args.no_structure,
        page_separator=args.page_separator
    )
    
    if args.file:
        converter.process_single_document(args.file)
    else:
        converter.convert_documents()


if __name__ == "__main__":
    main()