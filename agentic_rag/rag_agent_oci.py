from typing import List, Dict, Any, Optional
import json
import os
import argparse
import logging
from dotenv import load_dotenv

# OCI imports
import oci
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.llms.oci_generative_ai import OCIGenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Local imports
from agents.agent_factory import create_agents
try:
    from OracleDBVectorStore import OracleDBVectorStore
    ORACLE_DB_AVAILABLE = True
except ImportError:
    ORACLE_DB_AVAILABLE = False
    print("Oracle DB support not available. Install with: pip install oracledb sentence-transformers")
    


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PROFILE = "DEFAULT"  # Default OCI config profile
DEBUG = False  # Set to True for detailed debug output
SERVICE_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
MY_CONFIG_FILE_LOCATION = "~/.oci/config"
#
# load_oci_config(): load the OCI security config
#
def load_oci_config():
    # read OCI config to connect to OCI with API key
    oci_config = oci.config.from_file("~/.oci/config", CONFIG_PROFILE)

    # check the config to access to api keys
    if DEBUG:
        print("OCI Config:")
        print(oci_config)

    return oci_config

class OCIRAGAgent:
    def __init__(self, vector_store: OracleDBVectorStore, use_cot: bool = False, collection: str = None, skip_analysis: bool = False,
                 model_id: str = "cohere.command-r-plus-08-2024", compartment_id: str = None):
        """Initialize RAG agent with vector store and OCI Generative AI"""
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever()
        self.use_cot = use_cot
        self.collection = collection
        self.model_id = model_id
        self.compartment_id = compartment_id or os.getenv("OCI_COMPARTMENT_ID")
        
        # Set up OCI configuration
        config = load_oci_config()
        self.genai_client = OCIGenAI(
            auth_profile= CONFIG_PROFILE,
            auth_file_location= MY_CONFIG_FILE_LOCATION,
            model_id=model_id,
            compartment_id=self.compartment_id,
            service_endpoint=SERVICE_ENDPOINT,
            provider="cohere",
            model_kwargs={"temperature": 0, "max_tokens": 1500 }#, "stop": ["populous"]} # new endpoint
        )
        
        # Initialize specialized agents
        self.agents = create_agents(self.genai_client, vector_store, self.model_id, self.compartment_id) if use_cot else None
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using the agentic RAG pipeline"""
        logger.info(f"Processing query with collection: {self.collection}")
        
        # Process based on collection type and CoT setting
        if self.collection == "General Knowledge":
            # For General Knowledge, directly use general response
            if self.use_cot:
                return self._process_query_with_cot(query)
            else:
                return self._generate_general_response(query)
        else:
            # For PDF or Repository collections, use context-based processing
            if self.use_cot:
                return self._process_query_with_cot(query)
            else:
                return self._process_query_standard(query)
    
    def _process_query_with_cot(self, query: str) -> Dict[str, Any]:
        """Process query using Chain of Thought reasoning with multiple agents"""
        logger.info("Processing query with Chain of Thought reasoning")
        
        # Get initial context based on selected collection
        initial_context = []
        if self.collection == "PDF Collection":
            logger.info(f"Retrieving context from PDF Collection for query: '{query}'")
            pdf_context = self.vector_store.query_pdf_collection(query)
            initial_context.extend(pdf_context)
            logger.info(f"Retrieved {len(pdf_context)} chunks from PDF Collection")
            # Log each chunk with citation number but not full content
            for i, chunk in enumerate(pdf_context):
                source = chunk["metadata"].get("source", "Unknown")
                pages = chunk["metadata"].get("page_numbers", [])
                logger.info(f"Source [{i+1}]: {source} (pages: {pages})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        elif self.collection == "Repository Collection":
            logger.info(f"Retrieving context from Repository Collection for query: '{query}'")
            repo_context = self.vector_store.query_repo_collection(query)
            initial_context.extend(repo_context)
            logger.info(f"Retrieved {len(repo_context)} chunks from Repository Collection")
            for i, chunk in enumerate(repo_context):
                source = chunk["metadata"].get("source", "Unknown")
                file_path = chunk["metadata"].get("file_path", "Unknown")
                logger.info(f"Source [{i+1}]: {source} (file: {file_path})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        elif self.collection == "Web Knowledge Base":
            logger.info(f"Retrieving context from Web Knowledge Base for query: '{query}'")
            web_context = self.vector_store.query_web_collection(query)
            initial_context.extend(web_context)
            logger.info(f"Retrieved {len(web_context)} chunks from Web Knowledge Base")
            for i, chunk in enumerate(web_context):
                source = chunk["metadata"].get("source", "Unknown")
                title = chunk["metadata"].get("title", "Unknown")
                logger.info(f"Source [{i+1}]: {source} (title: {title})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        else:
            logger.info("Using General Knowledge collection, no context retrieval needed")
        
        try:
            # Step 1: Planning
            logger.info("Step 1: Planning")
            if not self.agents or "planner" not in self.agents:
                logger.warning("No planner agent available, using direct response")
                return self._generate_general_response(query)
            
            try:
                plan = self.agents["planner"].plan(query, initial_context)
                logger.info(f"Generated plan:\n{plan}")
            except Exception as e:
                logger.error(f"Error in planning step: {str(e)}")
                logger.info("Falling back to general response")
                return self._generate_general_response(query)
            
            # Step 2: Research each step (if researcher is available)
            logger.info("Step 2: Research")
            research_results = []
            if self.agents.get("researcher") is not None and initial_context:
                for step in plan.split("\n"):
                    if not step.strip():
                        continue
                    try:
                        step_research = self.agents["researcher"].research(query, step)
                        # Extract findings from research result
                        findings = step_research.get("findings", []) if isinstance(step_research, dict) else []
                        research_results.append({"step": step, "findings": findings})
                        
                        # Log which sources were used for this step
                        try:
                            source_indices = [initial_context.index(finding) + 1 for finding in findings if finding in initial_context]
                            logger.info(f"Research for step: {step}\nUsing sources: {source_indices}")
                        except ValueError as ve:
                            logger.warning(f"Could not find some findings in initial context: {str(ve)}")
                    except Exception as e:
                        logger.error(f"Error during research for step '{step}': {str(e)}")
                        research_results.append({"step": step, "findings": []})
            else:
                # If no researcher or no context, use the steps directly
                research_results = [{"step": step, "findings": []} for step in plan.split("\n") if step.strip()]
                logger.info("No research performed (no researcher agent or no context available)")
            
            # Step 3: Reasoning about each step
            logger.info("Step 3: Reasoning")
            if not self.agents.get("reasoner"):
                logger.warning("No reasoner agent available, using direct response")
                return self._generate_general_response(query)
            
            reasoning_steps = []
            for result in research_results:
                try:
                    step_reasoning = self.agents["reasoner"].reason(
                        query,
                        result["step"],
                        result["findings"] if result["findings"] else [{"content": "Using general knowledge", "metadata": {"source": "General Knowledge"}}]
                    )
                    reasoning_steps.append(step_reasoning)
                    logger.info(f"Reasoning for step: {result['step']}\n{step_reasoning}")
                except Exception as e:
                    logger.error(f"Error in reasoning for step '{result['step']}': {str(e)}")
                    reasoning_steps.append(f"Error in reasoning for this step: {str(e)}")
            
            # Step 4: Synthesize final answer
            logger.info("Step 4: Synthesis")
            if not self.agents.get("synthesizer"):
                logger.warning("No synthesizer agent available, using direct response")
                return self._generate_general_response(query)
            
            try:
                final_answer = self.agents["synthesizer"].synthesize(query, reasoning_steps)
                logger.info(f"Final synthesized answer:\n{final_answer}")
                
                # Handle string or dict response from synthesizer
                if isinstance(final_answer, str):
                    answer = final_answer
                else:
                    answer = final_answer.get("answer", final_answer)
                    
            except Exception as e:
                logger.error(f"Error in synthesis step: {str(e)}")
                logger.info("Falling back to general response")
                return self._generate_general_response(query)
            
            return {
                "answer": answer,
                "context": initial_context,
                "reasoning_steps": reasoning_steps
            }
        except Exception as e:
            logger.error(f"Error in CoT processing: {str(e)}", exc_info=True)
            logger.info("Falling back to general response")
            return self._generate_general_response(query)
    
    def _process_query_standard(self, query: str) -> Dict[str, Any]:
        """Process query using standard approach without Chain of Thought"""
        # Initialize context variables
        context = []
        
        # Get context based on selected collection
        if self.collection == "PDF Collection":
            logger.info(f"Retrieving context from PDF Collection for query: '{query}'")
            context = self.vector_store.query_pdf_collection(query)
            logger.info(f"Retrieved {len(context)} chunks from PDF Collection")
            for i, chunk in enumerate(context):
                source = chunk["metadata"].get("source", "Unknown")
                pages = chunk["metadata"].get("page_numbers", [])
                logger.info(f"Source [{i+1}]: {source} (pages: {pages})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        elif self.collection == "Repository Collection":
            logger.info(f"Retrieving context from Repository Collection for query: '{query}'")
            context = self.vector_store.query_repo_collection(query)
            logger.info(f"Retrieved {len(context)} chunks from Repository Collection")
            for i, chunk in enumerate(context):
                source = chunk["metadata"].get("source", "Unknown")
                file_path = chunk["metadata"].get("file_path", "Unknown")
                logger.info(f"Source [{i+1}]: {source} (file: {file_path})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        elif self.collection == "Web Knowledge Base":
            logger.info(f"Retrieving context from Web Knowledge Base for query: '{query}'")
            context = self.vector_store.query_web_collection(query)
            logger.info(f"Retrieved {len(context)} chunks from Web Knowledge Base")
            for i, chunk in enumerate(context):
                source = chunk["metadata"].get("source", "Unknown")
                title = chunk["metadata"].get("title", "Unknown")
                logger.info(f"Source [{i+1}]: {source} (title: {title})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        
        # Generate response using context if available, otherwise use general knowledge
        if context:
            logger.info(f"Generating response using {len(context)} context chunks")
            response = self._generate_response(query, context)
        else:
            logger.info("No context found, using general knowledge")
            response = self._generate_general_response(query)
        
        return response
    
    def _generate_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response based on the query and context using OCI Generative AI"""
        # Format context for the prompt
        formatted_context = "\n\n".join([f"Context {i+1}:\n{item['content']}" 
                                       for i, item in enumerate(context)])
        
        system_prompt = """You are an AI assistant answering questions based on the provided context.
Answer the question based on the context provided. If the answer is not in the context, say "I don't have enough information to answer this question." Be concise and accurate."""
        
        user_content = f"Context:\n{formatted_context}\n\nQuestion: {query}"
        
        prompt = PromptTemplate.from_template(user_content)
        chain = (
            prompt
            | self.genai_client
            | StrOutputParser()
)
        answer = chain.invoke({"query": query})
        
        # Add sources to response if available
        sources = {}
        if context:
            # Group sources by document
            for item in context:
                source = item['metadata'].get('source', 'Unknown')
                if source not in sources:
                    sources[source] = set()
                
                # Add page number if available
                if 'page' in item['metadata']:
                    sources[source].add(str(item['metadata']['page']))
                # Add file path if available for code
                if 'file_path' in item['metadata']:
                    sources[source] = item['metadata']['file_path']
            
            # Print concise source information
            print("\nSources detected:")
            for source, details in sources.items():
                if isinstance(details, set):  # PDF with pages
                    pages = ", ".join(sorted(details))
                    print(f"Document: {source} (pages: {pages})")
                else:  # Code with file path
                    print(f"Code file: {source}")
        
        return {
            "answer": answer,
            "context": context,
            "sources": sources
        }

    def _generate_general_response(self, query: str) -> Dict[str, Any]:
        """Generate a response using general knowledge when no context is available"""
        system_prompt = "You are a helpful AI assistant. Answer the following query using your general knowledge."
        user_content = f"Query: {query}\n\nAnswer:"
        
        prompt = PromptTemplate.from_template(user_content)
        chain = (
            {"input": RunnablePassthrough()}
            | prompt
            | self.genai_client
            | StrOutputParser()
        )
        answer = chain.invoke({"query": query})
        # Return a general response without context
        
        logger.info("No context available, using general knowledge response")    
        return {
            "answer": answer,
            "context": []
        }

def main():
    parser = argparse.ArgumentParser(description="Query documents using OCI Generative AI")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--store-path", default="chroma_db", help="Path to the vector store")
    parser.add_argument("--use-cot", action="store_true", help="Enable Chain of Thought reasoning")
    parser.add_argument("--collection", choices=["PDF Collection", "Repository Collection", "Web Knowledge Base", "General Knowledge"], 
                        help="Specify which collection to query")
    parser.add_argument("--model-id", default="cohere.command-r", help="OCI Gen AI model ID to use")
    parser.add_argument("--compartment-id", help="OCI compartment ID")
    parser.add_argument("--verbose", action="store_true", help="Show full content of sources")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check for OCI compartment ID
    compartment_id = args.compartment_id or os.getenv("OCI_COMPARTMENT_ID")
    if not compartment_id:
        print("✗ Error: OCI_COMPARTMENT_ID not found in environment variables or command line arguments")
        print("Please set the OCI_COMPARTMENT_ID environment variable or provide --compartment-id")
        exit(1)
    
    print("\nInitializing RAG agent...")
    print("=" * 50)
    
    try:
        store = OracleDBVectorStore()
            
        agent = OCIRAGAgent(
            store,
            use_cot=args.use_cot,
            collection=args.collection,
            model_id=args.model_id,
            compartment_id=compartment_id
        )
    
        
        print(f"\nProcessing query: {args.query}")
        print("=" * 50)
        
        response = agent.process_query(args.query)
        
        print("\nResponse:")
        print("-" * 50)
        print(response["answer"])
        
        if response.get("reasoning_steps"):
            print("\nReasoning Steps:")
            print("-" * 50)
            for i, step in enumerate(response["reasoning_steps"]):
                print(f"\nStep {i+1}:")
                print(step)
        
        if response.get("context"):
            print("\nSources used:")
            print("-" * 50)
            
            # Print concise list of sources
            for i, ctx in enumerate(response["context"]):
                source = ctx["metadata"].get("source", "Unknown")
                if "page_numbers" in ctx["metadata"]:
                    pages = ctx["metadata"].get("page_numbers", [])
                    print(f"[{i+1}] {source} (pages: {pages})")
                else:
                    file_path = ctx["metadata"].get("file_path", "Unknown")
                    print(f"[{i+1}] {source} (file: {file_path})")
                
                # Only print content if verbose flag is set
                if args.verbose:
                    content_preview = ctx["content"][:300] + "..." if len(ctx["content"]) > 300 else ctx["content"]
                    print(f"    Content: {content_preview}\n")
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()