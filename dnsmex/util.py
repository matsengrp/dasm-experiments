from pypdf import PdfWriter


def merge_pdfs(input_files, output_file):
    """Merge multiple PDF files into a single PDF.

    Args:
        input_files (list of str): Paths to the PDF files to merge.
        output_file (str): Path to the output merged PDF.

    Returns:
        None
    """
    merger = PdfWriter()
    for pdf in input_files:
        merger.append(pdf)
    merger.write(output_file)
    merger.close()
