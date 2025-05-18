import fitz  # PyMuPDF
import pandas as pd

# Actualizar y extraer texto deade  PDF
def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# Function to search for specific words and extract values
def search_for_values(text, keyword):
    lines = text.split('\n')  # Split text into lines
    results = []
    for line in lines:
        if keyword.lower() in line.lower():  # Case-insensitive search
            results.append(line)
    return results

# Main function
def main():
    # Insert the path to your PDF file
    pdf_path = input("Enter the path to your PDF file: ")
    pdf_text = extract_text_from_pdf(pdf_path)
    
    if pdf_text:
        print("PDF content extracted successfully!")
        # Ask the user for a keyword to search
        keyword = input("Enter the word or phrase to search for: ")
        matching_lines = search_for_values(pdf_text, keyword)
        
        if matching_lines:
            print(f"Found {len(matching_lines)} matching lines. Structuring into a table...")
            # Create a DataFrame from the matching lines
            df = pd.DataFrame(matching_lines, columns=["Matching Lines"])
            print(df)
            
            # Optionally, save the table to a CSV file
            save_csv = input("Do you want to save the table to a CSV file? (yes/no): ").strip().lower()
            if save_csv == 'yes':
                output_path = input("Enter the output CSV file path (e.g., output.csv): ")
                df.to_csv(output_path, index=False)
                print(f"Table saved to {output_path}")
        else:
            print(f"No matches found for '{keyword}'.")
    else:
        print("Failed to extract text from the PDF.")

if __name__ == "__main__":
    main()