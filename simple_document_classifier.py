import json
import re
import time
import os
from typing import Dict

class SimpleDocumentClassifier:
    """
    A lightweight document classifier that uses rule-based classification
    without heavy ML dependencies. This prevents system freezing and provides
    fast, reliable results.
    """
    
    def __init__(self):
        # Enhanced keyword sets for better accuracy
        self.bank_keywords = {
            # Core banking terms
            'account', 'balance', 'transaction', 'statement',
            'deposit', 'withdrawal', 'debit', 'credit',
            'bank', 'banking', 'compte', 'solde',
            
            # Transaction types
            'virement', 'transfer', 'retrait', 'versement',
            'prelevement', 'cheque', 'cb', 'carte',
            
            # Banking operations
            'operations', 'mouvement', 'releve', 'historique',
            'dab', 'atm', 'pos', 'tpe',
            
            # Amount indicators
            'eur', 'euro', 'usd', 'dollar', 'devise'
        }
        
        self.invoice_keywords = {
            # Invoice terms
            'invoice', 'facture', 'bill', 'devis',
            'quote', 'estimate', 'proforma',
            
            # Payment terms
            'payment', 'paiement', 'due', 'echeance',
            'amount', 'montant', 'total', 'subtotal',
            'sous-total', 'tva', 'tax', 'taxes',
            
            # Invoice structure
            'number', 'numero', 'date', 'client',
            'customer', 'vendor', 'supplier', 'fournisseur',
            
            # Financial terms
            'price', 'prix', 'cost', 'cout', 'tarif',
            'remise', 'discount', 'reduction'
        }
        
        # Patterns that strongly indicate document type
        self.bank_patterns = [
            r'relev[eé]\s+de\s+compte',  # French bank statement
            r'account\s+statement',       # English bank statement
            r'transaction\s+history',     # Transaction history
            r'solde\s+(?:créditeur|débiteur)',  # Account balance
            r'operations\s+du\s+compte',  # Account operations
            r'\d{2}/\d{2}/\d{4}\s*-\s*[A-Z\s]+\s*-\s*[+-]?\d+[.,]\d+',  # Transaction line
        ]
        
        self.invoice_patterns = [
            r'facture\s+n[°o]\s*\d+',     # Invoice number
            r'invoice\s+#?\d+',           # Invoice number
            r'montant\s+(?:ht|ttc)',      # Amount excluding/including tax
            r'total\s+(?:ht|ttc)',        # Total excluding/including tax
            r'tva\s+\d+[.,]\d*%',         # VAT percentage
            r'date\s+d[\'\u2019]?échéance',    # Due date
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        # Convert to lowercase for case-insensitive matching
        text = text.lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\n.,;:()%€$-]', ' ', text)
        
        return text.strip()
    
    def count_keyword_matches(self, text: str, keywords: set) -> int:
        """Count how many keywords appear in the text"""
        matches = 0
        words = re.findall(r'\b\w+\b', text)
        word_set = set(words)
        
        for keyword in keywords:
            if keyword in word_set:
                matches += 1
            # Also check for partial matches in the full text
            elif keyword in text:
                matches += 0.5
                
        return int(matches)
    
    def count_pattern_matches(self, text: str, patterns: list) -> int:
        """Count how many regex patterns match in the text"""
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        return matches
    
    def analyze_structure(self, text: str) -> Dict:
        """Analyze document structure for additional clues"""
        lines = text.split('\n')
        
        # Count different types of lines
        date_lines = sum(1 for line in lines if re.search(r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}', line))
        amount_lines = sum(1 for line in lines if re.search(r'[+-]?\d+[.,]\d{2}', line))
        header_lines = sum(1 for line in lines if len(line.strip()) > 0 and line.strip().isupper())
        
        return {
            'total_lines': len(lines),
            'date_lines': date_lines,
            'amount_lines': amount_lines,
            'header_lines': header_lines,
            'avg_line_length': sum(len(line) for line in lines) / max(len(lines), 1)
        }
    
    def classify_document(self, text: str) -> Dict:
        """
        Classify document using enhanced rule-based approach
        Returns classification with confidence and reasoning
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return {
                "document_type": "Unknown",
                "confidence": 0.0,
                "reason": "Empty document",
                "processing_time": 0.0
            }
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Count matches
        bank_keyword_score = self.count_keyword_matches(processed_text, self.bank_keywords)
        invoice_keyword_score = self.count_keyword_matches(processed_text, self.invoice_keywords)
        
        bank_pattern_score = self.count_pattern_matches(processed_text, self.bank_patterns)
        invoice_pattern_score = self.count_pattern_matches(processed_text, self.invoice_patterns)
        
        # Analyze structure
        structure = self.analyze_structure(processed_text)
        
        # Calculate weighted scores
        bank_total = (bank_keyword_score * 1.0) + (bank_pattern_score * 2.0)
        invoice_total = (invoice_keyword_score * 1.0) + (invoice_pattern_score * 2.0)
        
        # Structural bonuses
        if structure['date_lines'] > 3 and structure['amount_lines'] > 3:
            bank_total += 1.0  # Likely transaction list
        
        if structure['header_lines'] > 0 and 'total' in processed_text:
            invoice_total += 1.0  # Likely invoice structure
        
        # Determine classification
        processing_time = time.time() - start_time
        
        if bank_total > invoice_total and bank_total > 0:
            confidence = min(bank_total / (bank_total + invoice_total + 1), 0.95)
            return {
                "document_type": "Bank Statement",
                "confidence": round(confidence, 2),
                "reason": f"Bank keywords: {bank_keyword_score}, patterns: {bank_pattern_score}",
                "processing_time": round(processing_time, 3),
                "scores": {
                    "bank_total": bank_total,
                    "invoice_total": invoice_total,
                    "structure": structure
                }
            }
        elif invoice_total > bank_total and invoice_total > 0:
            confidence = min(invoice_total / (bank_total + invoice_total + 1), 0.95)
            return {
                "document_type": "Invoice",
                "confidence": round(confidence, 2),
                "reason": f"Invoice keywords: {invoice_keyword_score}, patterns: {invoice_pattern_score}",
                "processing_time": round(processing_time, 3),
                "scores": {
                    "bank_total": bank_total,
                    "invoice_total": invoice_total,
                    "structure": structure
                }
            }
        else:
            return {
                "document_type": "Unknown",
                "confidence": 0.0,
                "reason": "Insufficient indicators for classification",
                "processing_time": round(processing_time, 3),
                "scores": {
                    "bank_total": bank_total,
                    "invoice_total": invoice_total,
                    "structure": structure
                }
            }

def process_file(file_path: str) -> Dict:
    """Process a file and return classification results"""
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        # Check file size (limit to 10MB for safety)
        file_size = os.path.getsize(file_path)
        if file_size > 10 * 1024 * 1024:
            return {"error": f"File too large: {file_size / (1024*1024):.1f} MB"}
        
        # Read file with error handling
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        if not content.strip():
            return {"error": "File is empty"}
        
        # Classify document
        classifier = SimpleDocumentClassifier()
        result = classifier.classify_document(content)
        
        # Add file metadata
        result.update({
            "file_name": os.path.basename(file_path),
            "file_size_kb": round(file_size / 1024, 1),
            "content_length": len(content),
            "method": "rule_based"
        })
        
        return result
        
    except Exception as e:
        return {
            "file_name": os.path.basename(file_path) if file_path else "unknown",
            "error": f"Processing failed: {str(e)}"
        }

def create_sample_files():
    """Create sample files for testing if they don't exist"""
    os.makedirs("output", exist_ok=True)
    
    # Sample bank statement
    bank_content = """
RELEVE DE COMPTE
Banque: Société Générale
Compte: 0082348808
Période: 01/09/2024 au 30/09/2024

OPERATIONS DU COMPTE:
05/09/2024 - VIREMENT ENTRANT SALAIRE - +2500.00 EUR
04/09/2024 - ACHAT CB SUPERMARCHE LECLERC - -85.50 EUR
03/09/2024 - RETRAIT DAB PLACE REPUBLIQUE - -100.00 EUR
02/09/2024 - PRELEVEMENT EDF ELECTRICITE - -89.30 EUR
01/09/2024 - CHEQUE N°1234567 - -250.00 EUR

SOLDE CREDITEUR: 2314.50 EUR
"""
    
    # Sample invoice
    invoice_content = """
FACTURE N° 2024-001
Date: 15/09/2024
Date d'échéance: 15/10/2024

Client: Entreprise ABC
Adresse: 123 Rue de la Paix, 75001 Paris

PRESTATIONS:
- Consultation informatique (5h) - 300.00 EUR
- Développement logiciel (10h) - 600.00 EUR

SOUS-TOTAL HT: 900.00 EUR
TVA 20%: 180.00 EUR
TOTAL TTC: 1080.00 EUR

Modalités de paiement: Virement bancaire
"""
    
    # Write sample files
    with open("output/sample_bank_statement.txt", "w", encoding="utf-8") as f:
        f.write(bank_content)
    
    with open("output/sample_invoice.txt", "w", encoding="utf-8") as f:
        f.write(invoice_content)
    
    print("Sample files created:")
    print("- output/sample_bank_statement.txt")
    print("- output/sample_invoice.txt")

def main():
    """Main function with comprehensive testing"""
    print("Simple Document Classifier")
    print("=" * 50)
    print("This classifier uses rule-based analysis and does NOT require")
    print("heavy ML libraries that could freeze your system.\n")
    
    # Create sample files if needed
    create_sample_files()
    
    # Test files
    test_files = [
        "output/0dbef502-RELEVES_0082348808_20240906_page1.txt",
        "output/sample_bank_statement.txt",
        "output/sample_invoice.txt"
    ]
    
    # Process each file
    total_start = time.time()
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\nProcessing: {file_path}")
            print("-" * 40)
            
            result = process_file(file_path)
            
            # Pretty print results
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\nFile not found: {file_path}")
    
    total_time = time.time() - total_start
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print("\nClassification complete! No system freezing risk.")

if __name__ == "__main__":
    main()