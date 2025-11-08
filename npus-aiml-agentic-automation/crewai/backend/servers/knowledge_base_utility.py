import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import hashlib
from datetime import datetime

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain.schema import Document
from custom_llm.nestle_llm import NestleLLM
from custom_llm.nestle_embeddings import NestleEmbeddingsLangChain
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""

    model_name: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "mlops_knowledge_base"
    persist_directory: str = "./chroma_db"
    index_file: str = "embeddings_index.json"


class MLOpsKnowledgeBase:
    """
    A RAG-based knowledge base for MLOps documentation.
    Supports independent embedding generation and retrieval operations with incremental updates.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None, llm=None):
        """
        Initialize the knowledge base.

        Args:
            config: Embedding configuration
            llm: Custom LLM instance (NestleLLM)
        """
        self.config = config or EmbeddingConfig()
        self.llm = llm
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.index_path = (
            self._resolve_path(self.config.persist_directory) / self.config.index_file
        )

    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a path to an absolute path.
        If the path is relative, join it with the current working directory.

        Args:
            path: Input path (relative or absolute)

        Returns:
            Resolved absolute Path object
        """
        path_obj = Path(path)

        # If path is already absolute, return it
        if path_obj.is_absolute():
            return path_obj

        # Otherwise, join with current working directory
        resolved = Path.cwd() / path_obj
        return resolved.resolve()

    def _calculate_file_checksum(self, file_path: str) -> str:
        """
        Calculate MD5 checksum of a file.

        Args:
            file_path: Path to the file

        Returns:
            MD5 checksum as hex string
        """
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _load_index(self) -> Dict[str, any]:
        """
        Load the embeddings index file.

        Returns:
            Dictionary with indexed file information
        """
        if not self.index_path.exists():
            return {
                "version": "1.0",
                "last_updated": None,
                "config": {
                    "model_name": self.config.model_name,
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                },
                "files": {},
            }

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load index file: {e}")
            return {
                "version": "1.0",
                "last_updated": None,
                "config": {
                    "model_name": self.config.model_name,
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                },
                "files": {},
            }

    def _save_index(self, index_data: Dict[str, any]):
        """
        Save the embeddings index file.

        Args:
            index_data: Index data to save
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

        print(f"Index saved to: {self.index_path}")

    def _check_config_changed(self, index_data: Dict[str, any]) -> bool:
        """
        Check if embedding configuration has changed.

        Args:
            index_data: Current index data

        Returns:
            True if config changed, False otherwise
        """
        old_config = index_data.get("config", {})

        return (
            old_config.get("model_name") != self.config.model_name
            or old_config.get("chunk_size") != self.config.chunk_size
            or old_config.get("chunk_overlap") != self.config.chunk_overlap
        )

    def _identify_changes(
        self, file_paths: List[str], index_data: Dict[str, any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Identify new, modified, and deleted files.

        Args:
            file_paths: Current list of file paths
            index_data: Existing index data

        Returns:
            Tuple of (new_files, modified_files, deleted_files)
        """
        indexed_files = index_data.get("files", {})
        current_files_set = set(file_paths)
        indexed_files_set = set(indexed_files.keys())

        # New files
        new_files = list(current_files_set - indexed_files_set)

        # Potentially modified files
        common_files = current_files_set & indexed_files_set
        modified_files = []

        for file_path in common_files:
            current_checksum = self._calculate_file_checksum(file_path)
            indexed_checksum = indexed_files[file_path].get("checksum")

            if current_checksum != indexed_checksum:
                modified_files.append(file_path)

        # Deleted files
        deleted_files = list(indexed_files_set - current_files_set)

        return new_files, modified_files, deleted_files

    def _initialize_embeddings(self):
        """Initialize the embedding model (lazy loading)"""
        if self.embeddings is None:
            self.embeddings = NestleEmbeddingsLangChain(
                model=self.config.model_name,
                client_id=os.getenv("NESTLE_CLIENT_ID"),
                client_secret=os.getenv("NESTLE_CLIENT_SECRET"),
            )

    def _initialize_vectorstore(self):
        """Initialize the vector store connection"""
        if self.vectorstore is None:
            self._initialize_embeddings()

            # Resolve persist directory path
            persist_path = self._resolve_path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(anonymized_telemetry=False),
            )

            # Create or get collection
            self.vectorstore = Chroma(
                client=client,
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(persist_path),
            )

    def _delete_file_chunks(self, file_path: str):
        """
        Delete all chunks associated with a specific file from the vector store.

        Args:
            file_path: Path of the file whose chunks should be deleted
        """
        self._initialize_vectorstore()

        # Query for all chunks with this source file
        collection = self.vectorstore._collection
        results = collection.get(where={"source": file_path})

        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            print(
                f"Deleted {len(results['ids'])} chunks for file: {Path(file_path).name}"
            )

    # ==================== EMBEDDING FLOW ====================

    def discover_markdown_files(self, root_directory: str) -> List[str]:
        """
        Walk through directory to find all markdown files.

        Args:
            root_directory: Root directory to search (relative or absolute)

        Returns:
            List of absolute paths to markdown files
        """
        # Resolve the root directory path
        root_path = self._resolve_path(root_directory)

        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {root_path}")

        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {root_path}")

        markdown_files = []

        # Walk through directory
        for file_path in root_path.rglob("*.md"):
            if file_path.is_file():
                markdown_files.append(str(file_path.absolute()))

        print(f"Discovered {len(markdown_files)} markdown files in {root_path}")
        return sorted(markdown_files)

    def load_markdown_content(self, file_paths: List[str]) -> List[Document]:
        """
        Load content from markdown files.

        Args:
            file_paths: List of file paths to load

        Returns:
            List of Document objects with content and metadata
        """
        documents = []

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Calculate checksum
                checksum = self._calculate_file_checksum(file_path)

                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "file_hash": checksum,
                    },
                )
                documents.append(doc)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        print(f"Loaded {len(documents)} documents")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for embedding.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def create_embeddings(
        self, root_directory: str, batch_size: int = 100, force_rebuild: bool = False
    ) -> Dict[str, any]:
        """
        Complete embedding pipeline with incremental updates.
        Only processes new or modified files unless force_rebuild is True.

        Args:
            root_directory: Root directory containing markdown files (relative or absolute)
            batch_size: Number of documents to process in each batch
            force_rebuild: If True, rebuild entire index from scratch

        Returns:
            Dictionary with statistics about the embedding process
        """
        resolved_path = self._resolve_path(root_directory)
        print(f"Starting embedding process for directory: {resolved_path}")

        # Load existing index
        index_data = self._load_index()

        # Check if config changed
        if self._check_config_changed(index_data) and not force_rebuild:
            print("⚠️  Configuration changed! Rebuilding entire index...")
            force_rebuild = True

        # Discover files
        file_paths = self.discover_markdown_files(root_directory)
        if not file_paths:
            return {"error": "No markdown files found", "files_processed": 0}

        if force_rebuild:
            print("🔄 Force rebuild enabled - processing all files")
            new_files = file_paths
            modified_files = []
            deleted_files = []

            # Clear existing collection
            self._initialize_vectorstore()
            try:
                self.vectorstore.delete_collection()
                self.vectorstore = None
                self._initialize_vectorstore()
            except:
                pass

            # Reset index
            index_data = {
                "version": "1.0",
                "last_updated": None,
                "config": {
                    "model_name": self.config.model_name,
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                },
                "files": {},
            }
        else:
            # Identify changes
            new_files, modified_files, deleted_files = self._identify_changes(
                file_paths, index_data
            )

            print(f"\n📊 Change Summary:")
            print(f"  ✅ New files: {len(new_files)}")
            print(f"  🔄 Modified files: {len(modified_files)}")
            print(f"  ❌ Deleted files: {len(deleted_files)}")

            if not new_files and not modified_files and not deleted_files:
                print("\n✨ No changes detected - embeddings are up to date!")
                return {
                    "status": "up_to_date",
                    "files_discovered": len(file_paths),
                    "new_files": 0,
                    "modified_files": 0,
                    "deleted_files": 0,
                    "embedding_model": self.config.model_name,
                    "collection_name": self.config.collection_name,
                    "source_directory": str(resolved_path),
                }

            # Handle deleted files
            for file_path in deleted_files:
                self._delete_file_chunks(file_path)
                del index_data["files"][file_path]

            # Handle modified files (delete old chunks)
            for file_path in modified_files:
                self._delete_file_chunks(file_path)

        # Process new and modified files
        files_to_process = new_files + modified_files

        if files_to_process:
            print(f"\n🔨 Processing {len(files_to_process)} files...")

            # Load and chunk documents
            documents = self.load_markdown_content(files_to_process)
            if not documents:
                return {"error": "No documents loaded", "files_processed": 0}

            chunks = self.chunk_documents(documents)

            # Initialize vectorstore
            self._initialize_vectorstore()

            # Add documents to vectorstore in batches
            total_chunks = len(chunks)
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i : i + batch_size]
                self.vectorstore.add_documents(batch)
                print(
                    f"  Processed batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}"
                )

            # Update index for processed files
            for file_path in files_to_process:
                checksum = self._calculate_file_checksum(file_path)
                index_data["files"][file_path] = {
                    "checksum": checksum,
                    "last_processed": datetime.now().isoformat(),
                    "filename": Path(file_path).name,
                }

        # Update index metadata
        index_data["last_updated"] = datetime.now().isoformat()
        index_data["config"] = {
            "model_name": self.config.model_name,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }

        # Save index
        self._save_index(index_data)

        stats = {
            "status": "updated" if not force_rebuild else "rebuilt",
            "files_discovered": len(file_paths),
            "new_files": len(new_files),
            "modified_files": len(modified_files),
            "deleted_files": len(deleted_files),
            "total_indexed_files": len(index_data["files"]),
            "chunks_processed": len(chunks) if files_to_process else 0,
            "embedding_model": self.config.model_name,
            "collection_name": self.config.collection_name,
            "source_directory": str(resolved_path),
        }

        print(f"\n✅ Embedding complete: {stats}")
        return stats

    def update_embeddings(
        self, root_directory: str, force_update: bool = False
    ) -> Dict[str, any]:
        """
        Update embeddings for new or modified files.
        This is now a wrapper around create_embeddings with smart incremental updates.

        Args:
            root_directory: Root directory containing markdown files (relative or absolute)
            force_update: If True, re-embed all documents

        Returns:
            Dictionary with update statistics
        """
        return self.create_embeddings(root_directory, force_rebuild=force_update)

    # ==================== RETRIEVAL FLOW ====================

    def retrieve_relevant_chunks(
        self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of (Document, similarity_score) tuples
        """
        self._initialize_vectorstore()

        # Perform similarity search with scores
        results = self.vectorstore.similarity_search_with_score(
            query=query, k=k, filter=filter_metadata
        )

        return results

    def format_context(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """
        Format retrieved documents into context string.

        Args:
            retrieved_docs: List of (Document, score) tuples

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, (doc, score) in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("filename", "Unknown")
            content = doc.page_content.strip()
            context_parts.append(
                f"[Source {i}: {source} (Relevance: {score:.3f})]\n{content}\n"
            )

        return "\n---\n".join(context_parts)

    def generate_answer(
        self, query: str, k: int = 5, temperature: float = 0.7, max_tokens: int = 1000
    ) -> Dict[str, any]:
        """
        Generate answer using RAG: retrieve context and generate response with LLM.

        Args:
            query: User question
            k: Number of documents to retrieve
            temperature: LLM temperature
            max_tokens: Maximum tokens for response

        Returns:
            Dictionary with answer, sources, and metadata
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Please provide an LLM instance.")

        # Retrieve relevant chunks
        retrieved_docs = self.retrieve_relevant_chunks(query, k=k)

        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "sources": [],
                "query": query,
            }

        # Format context
        context = self.format_context(retrieved_docs)

        # Create prompt
        prompt = f"""You are an MLOps expert assistant. Answer the following question based on the provided context from the MLOps knowledge base.

Context:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- If the context doesn't contain enough information, say so
- Reference specific sources when applicable
- Be concise but thorough

Answer:"""

        # Generate response using custom LLM
        try:
            response = self.llm.call(
                prompt, temperature=temperature, max_tokens=max_tokens
            )

            # Extract sources
            sources = [
                {
                    "filename": doc.metadata.get("filename"),
                    "source": doc.metadata.get("source"),
                    "relevance_score": float(score),
                }
                for doc, score in retrieved_docs
            ]

            return {
                "answer": response,
                "sources": sources,
                "query": query,
                "num_sources": len(sources),
            }

        except Exception as e:
            return {"error": f"Error generating answer: {str(e)}", "query": query}

    def search_knowledge_base(
        self, query: str, k: int = 10, include_scores: bool = True
    ) -> List[Dict[str, any]]:
        """
        Search the knowledge base without LLM generation (pure retrieval).

        Args:
            query: Search query
            k: Number of results
            include_scores: Include similarity scores

        Returns:
            List of search results with metadata
        """
        retrieved_docs = self.retrieve_relevant_chunks(query, k=k)

        results = []
        for doc, score in retrieved_docs:
            result = {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename"),
                "source": doc.metadata.get("source"),
            }
            if include_scores:
                result["relevance_score"] = float(score)
            results.append(result)

        return results

    # ==================== UTILITY METHODS ====================

    def get_collection_stats(self) -> Dict[str, any]:
        """Get statistics about the vector database collection"""
        self._initialize_vectorstore()

        collection = self.vectorstore._collection
        count = collection.count()

        # Load index data
        index_data = self._load_index()

        return {
            "collection_name": self.config.collection_name,
            "total_chunks": count,
            "total_indexed_files": len(index_data.get("files", {})),
            "last_updated": index_data.get("last_updated"),
            "persist_directory": self.config.persist_directory,
            "embedding_model": self.config.model_name,
            "config": index_data.get("config", {}),
        }

    def get_indexed_files(self) -> List[Dict[str, any]]:
        """
        Get list of all indexed files with their metadata.

        Returns:
            List of file information dictionaries
        """
        index_data = self._load_index()

        files_info = []
        for file_path, file_data in index_data.get("files", {}).items():
            files_info.append(
                {
                    "path": file_path,
                    "filename": file_data.get("filename"),
                    "checksum": file_data.get("checksum"),
                    "last_processed": file_data.get("last_processed"),
                }
            )

        return sorted(files_info, key=lambda x: x["filename"])

    def clear_database(self):
        """Clear the entire vector database and index"""
        self._initialize_vectorstore()
        self.vectorstore.delete_collection()
        self.vectorstore = None

        # Delete index file
        if self.index_path.exists():
            self.index_path.unlink()

        print("Vector database and index cleared")


if __name__ == "__main__":

    # Initialize the custom LLM
    def initialize_llm():
        """Initialize the Nestle LLM with credentials from environment"""
        client_id = os.getenv("NESTLE_CLIENT_ID")
        client_secret = os.getenv("NESTLE_CLIENT_SECRET")
        model = os.getenv("NESTLE_MODEL", "gpt-4.1")

        # Create LLM for CrewAI
        llm = NestleLLM(
            model=model,
            client_id=client_id,
            client_secret=client_secret,
        )

        return llm

    # Initialize the LLM once at module level
    llm = initialize_llm()

    # Example configuration
    config = EmbeddingConfig(
        chunk_size=800, chunk_overlap=150, persist_directory=os.getenv("VECTOR_DB_LOC")
    )

    # Step 1: Create embeddings (first time)
    # This will process all files and create the index
    kb = MLOpsKnowledgeBase(config)
    stats = kb.create_embeddings(os.getenv("WIKI_REPO_LOC"))
    print("\nFirst run stats:", json.dumps(stats, indent=2))

    # Step 2: Check collection stats
    collection_stats = kb.get_collection_stats()
    print("\nCollection stats:", json.dumps(collection_stats, indent=2))

    # Step 3: Get indexed files
    indexed_files = kb.get_indexed_files()
    print(f"\nIndexed files: {len(indexed_files)}")

    # Step 4: Use for retrieval
    kb_retrieval = MLOpsKnowledgeBase(config, llm=llm)
    answer = kb_retrieval.generate_answer(
        "What are the different workflow files in MLOPs template"
    )
    print("\nAnswer:", json.dumps(answer, indent=2))
