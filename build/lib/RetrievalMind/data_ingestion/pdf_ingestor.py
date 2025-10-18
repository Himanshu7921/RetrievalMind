from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader

class PDFDocumentIngestor:
    """
    Handles loading of PDF documents using LangChain's supported loaders.

    This class provides a unified interface to load PDFs either with:
      - `PyMuPDFLoader` (fast and efficient using the PyMuPDF backend)
      - `PyPDFLoader` (standard loader for compatibility)

    Attributes:
        file_path (str): Path to the PDF file to be loaded.
        encoding (str): File encoding, defaults to 'utf-8'.
        loader_type (str): Specifies which loader to use.
                           Use 'mu' for PyMuPDFLoader or 'std' for PyPDFLoader.
    """

    def __init__(self, file_path: str, encoding: str = 'utf-8', loader_type: str = 'mu'):
        """
        Initializes the PDFDocumentIngestor with file path, encoding, and loader type.

        Args:
            file_path (str): The path to the PDF file.
            encoding (str): Encoding format for the file (default: 'utf-8').
            loader_type (str): The loader to use — 'mu' (PyMuPDFLoader) or 'std' (PyPDFLoader).
        """
        self.file_path = file_path
        self.encoding = encoding
        self.loader_type = loader_type.lower()

    def load_document(self):
        """
        Loads the PDF using the specified loader.

        Returns:
            A LangChain-compatible document loader instance.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            TypeError: If the provided loader type is invalid.
            Exception: For any other unexpected errors during loading.
        """
        try:
            # Select loader based on loader_type
            if self.loader_type == "mu":
                # Use PyMuPDFLoader for faster performance
                return PyMuPDFLoader(self.file_path)

            elif self.loader_type == "std":
                # Use standard PyPDFLoader for general compatibility
                return PyPDFLoader(self.file_path)

            else:
                # Raise explicit error if the loader type is not recognized
                raise TypeError(
                    "Invalid loader type. Use 'mu' for PyMuPDFLoader or 'std' for PyPDFLoader."
                )

        except FileNotFoundError:
            # Raised when the given file path does not exist
            raise FileNotFoundError(f"File not found: {self.file_path}")

        except TypeError as e:
            # Propagate invalid loader type error
            raise e

        except Exception as e:
            # Catch-all for unexpected issues (e.g., file corruption, permission errors)
            raise RuntimeError(f"An unexpected error occurred while loading the PDF: {e}")