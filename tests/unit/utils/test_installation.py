import unittest
import importlib
import sys

class TestInstallation(unittest.TestCase):
    """Test the installation of the sophos_rag package."""

    def test_package_import(self):
        """Test that the package can be imported."""
        try:
            import sophos_rag
            self.assertIsNotNone(sophos_rag)
            print(f"Successfully imported sophos_rag version {sophos_rag.__version__}")
        except ImportError as e:
            self.fail(f"Failed to import sophos_rag: {e}")

    def test_core_modules(self):
        """Test that all core modules can be imported."""
        core_modules = [
            'sophos_rag.cli',
            'sophos_rag.embeddings',
            'sophos_rag.generator',
            'sophos_rag.pipeline',
            'sophos_rag.retriever',
            'sophos_rag.utils'
        ]
        
        for module in core_modules:
            try:
                imported_module = importlib.import_module(module)
                print(f"Successfully imported {module}")
                self.assertIsNotNone(imported_module)
            except ImportError as e:
                self.fail(f"Failed to import {module}: {e}")

    def test_numpy_import(self):
        """Test that numpy can be imported."""
        try:
            import numpy
            print(f"Successfully imported numpy version {numpy.__version__}")
            self.assertTrue(True, "numpy is installed")
        except ImportError as e:
            self.fail(f"Failed to import numpy: {e}")

if __name__ == '__main__':
    unittest.main() 