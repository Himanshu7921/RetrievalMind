from langchain_community.document_loaders import TextLoader

class TextDocumentIngestor:
    """
    Handles loading of plain text documents using LangChain's TextLoader.

    This class provides a simple and consistent interface for ingesting
    text-based files into LangChain-compatible document objects.

    Attributes:
        file_path (str): Path to the text file to be loaded.
        encoding (str): File encoding, defaults to 'utf-8'.
    """

    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        """
        Initializes the TextDocumentIngestor with file path and encoding.

        Args:
            file_path (str): The path to the text file.
            encoding (str): Encoding format for the file (default: 'utf-8').
        """
        self.file_path = file_path
        self.encoding = encoding

    def load_document(self):
        """
        Loads the text file using LangChain's TextLoader.

        Returns:
            TextLoader: A LangChain-compatible document loader instance for text files.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
        """
        # Basic validation for file existence
        try:
            # Create and return the loader instance
            return TextLoader(self.file_path, encoding=self.encoding)
        except FileNotFoundError:
            # Raise an explicit error for missing files
            raise FileNotFoundError(f"File not found: {self.file_path}")