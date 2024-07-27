from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone as PineconeClient
import os
from dotenv import load_dotenv

class ChatBot:
    def __init__(self):
        load_dotenv()

        # Load and split documents
        self.loader = TextLoader('Chatbot/Is Bitcoin the Future of Money?')
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)

        # Debugging to inspect the attributes of the Document object
        print("Inspecting Document attributes...")
        if self.docs:
            print(self.docs[0].__dict__)  # Print attributes of the first document

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Initialize Pinecone client
        api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_client = PineconeClient(api_key=api_key)

        self.index_name = "langchain-demo"

        # Check if the index exists
        index_list = self.pinecone_client.list_indexes()
        if self.index_name not in index_list:
            # Create the index with default parameters
            try:
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=768,  # Set the dimension based on your embeddings
                    metric='cosine'  # You can use other metrics like 'euclidean' if needed
                )
                print("Index created successfully.")
            except Exception as e:
                print(f"Error occurred while creating index: {e}")
        
        self.index = self.pinecone_client.Index(self.index_name)

        # Index documents
        if not self.pinecone_client.describe_index(self.index_name).get("status") == "ready":
            # Convert documents to the format expected by Pinecone
            index_data = [{'id': str(i), 'values': self.embeddings.embed_documents([doc.page_content])[0]} for i, doc in enumerate(self.docs)]
            try:
                self.index.upsert(vectors=index_data)
                print("Documents indexed successfully.")
            except Exception as e:
                print(f"Error occurred while indexing documents: {e}")

        # Set up the LLM
        self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id, 
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
            temperature=0.8,  # Directly specifying parameters
            top_k=50
        )

        # Define prompt template
        template = """
        You are a knowledgeable expert on bitcoin. These Humans will ask you questions about the article stored. Use the following piece of context to answer the question. 
        If you don't know the answer, just say you don't know. 
        Your answer should be short and concise, no longer than two sentences.

        Context: {context}
        Question: {question}
        Answer: 
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def get_response(self, question):
        print(f"Received question: {question}")

        try:
            # Convert the question into a vector
            query_vector = self.embeddings.embed_documents([question])[0]

            # Perform the search
            query_result = self.index.query(
                vector=query_vector,  # Use the vectorized query here
                top_k=5  # Adjust top_k as needed
            )

            # Print the query result to inspect its format
            print("Query result:", query_result)

            # Extract relevant results
            if 'matches' in query_result:
                matches = query_result['matches']
                if matches:
                    # Attempt to extract context from each match
                    # Adjust this based on the actual structure of 'matches'
                    context = " ".join([str(match.get('values', '')) for match in matches])
                else:
                    context = "No relevant context found."
            else:
                context = "No relevant context found."

            # Format the prompt
            formatted_prompt = self.prompt.format(context=context, question=question)
            print(f"Formatted prompt: {formatted_prompt}")

            # Call the LLM
            response = self.llm(formatted_prompt)
            print(f"Response from LLM: {response}")
            return response

        except Exception as e:
            print(f"Error occurred: {e}")
            return "An error occurred while generating the response."

# CLI script to interact with the bot
if __name__ == "__main__":
    bot = ChatBot()

    print("Welcome to the World of Bitcoin!")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        response = bot.get_response(user_input)
        print(f"Bot: {response}")
