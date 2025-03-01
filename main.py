import streamlit as st
import os
import time
import pandas as pd
from openai import OpenAI
from astrapy import DataAPIClient

# Set page configuration
st.set_page_config(
    page_title="Credit Card Offers Assistant",
    page_icon="💳",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'page' not in st.session_state:
    st.session_state.page = 'name_input'
if 'name' not in st.session_state:
    st.session_state.name = ''
if 'selected_brands' not in st.session_state:
    st.session_state.selected_brands = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Define the brands extracted from your data
brands = [
    "Acropolispizzapasta",
    "Acropolispizzapastaeverett",
    "Addeo's Of The Bronx",
    "Addeo's of the Bronx",
    "Adelle's",
    "adidas.com",
    "adidasadidas",
    "Adobe",
    "Adorama",
    "Ado's Kitchen & Bar",
    "ADT",
    "AÃ±ejo Tequila Joint",
    "Aera Smart Home Fragrance",
    "Aeropostale",
    "Aesop",
    "Afchomeclub",
    "African Cuisine",
    "African Soul Food"
]

# Astra DB and OpenAI setup functions

def get_openai_client():
    """Initialize and return the OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", None))
    if not api_key:
        st.error("OpenAI API key not found. Please set it as an environment variable or in Streamlit secrets.")
        return None
    return OpenAI(api_key=api_key)

def get_astra_collection():
    """Connect to Astra DB and return the collection."""
    try:
        app_token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN", st.secrets.get("ASTRA_DB_APPLICATION_TOKEN", None))
        api_endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT", st.secrets.get("ASTRA_DB_API_ENDPOINT", None))
        
        if not app_token or not api_endpoint:
            st.error("Astra DB credentials not found. Please set them as environment variables or in Streamlit secrets.")
            return None
        
        astra_client = DataAPIClient(app_token)
        database = astra_client.get_database(api_endpoint)
        collection = database.get_collection(name="openai", keyspace="default_keyspace")
        return collection
    except Exception as e:
        st.error(f"Failed to connect to Astra DB: {e}")
        return None

def save_preferences_to_astra(name, selected_brands):
    """Save user preferences to Astra DB."""
    try:
        collection = get_astra_collection()
        if not collection:
            return False
        
        # Create a document with user preferences
        doc = {
            "name": name,
            "brands": selected_brands,
            "timestamp": time.time()
        }
        
        # Insert the document
        result = collection.create_document(doc)
        return True if result else False
    except Exception as e:
        st.error(f"Failed to save preferences: {e}")
        return False

def get_query_embedding(query):
    """Generate embedding for query text."""
    client = get_openai_client()
    if not client:
        return None
    
    try:
        embedding_response = client.embeddings.create(
            input=query, 
            model="text-embedding-3-small"
        )
        return embedding_response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def retrieve_documents(query, brands=None, top_k=5):
    """Retrieve relevant documents from Astra DB based on query."""
    collection = get_astra_collection()
    if not collection:
        return []
    
    query_embedding = get_query_embedding(query)
    if not query_embedding:
        return []
    
    try:
        # Create filter if brands are specified
        filter_condition = {}
        if brands and len(brands) > 0:
            filter_condition = {"brand": {"$in": brands}}
        
        # Perform vector search
        results = collection.find(
            filter=filter_condition if filter_condition else None,
            sort={"$vector": query_embedding},
            limit=top_k
        )
        
        return results if results else []
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

def generate_answer(query, documents):
    """Generate response based on retrieved documents using OpenAI."""
    client = get_openai_client()
    if not client or not documents:
        return "I couldn't find relevant information or there was an issue with the service."
    
    # Combine document content to create context
    context = "\n\n".join([doc.get("content", "") for doc in documents])
    
    # Create prompt with the retrieved context
    prompt = f"""
You are an AI assistant helping users find information about credit card offers and rewards.
Answer the user's question using ONLY the following information from the database:

{context}

If the information needed to answer the question is not in the provided context, say "I don't have that information in my database."
Be concise and helpful. Format any offers nicely.

User question: {query}
"""
    
    try:
        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o",  # You can change this to another model
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in credit card offers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more focused responses
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "I encountered an issue while generating a response. Please try again."

# Page functions

def name_input_page():
    """First page to collect user's name."""
    st.title("Welcome to Credit Card Offers Assistant")
    st.subheader("Let's get to know you")
    
    name = st.text_input("What's your name?", value=st.session_state.name)
    
    if st.button("Continue"):
        if name.strip():
            st.session_state.name = name
            st.session_state.page = 'brand_selection'
            st.rerun()
        else:
            st.error("Please enter your name to continue.")

def brand_selection_page():
    """Second page to select brand preferences."""
    st.title(f"Hi {st.session_state.name}, Select Your Preferred Brands")
    st.subheader("Choose the brands you're interested in")
    
    # Create a multiselect for brand selection
    selected_brands = st.multiselect(
        "Select brands you prefer:",
        options=brands,
        default=st.session_state.selected_brands
    )
    
    if st.button("Save Preferences"):
        if selected_brands:
            st.session_state.selected_brands = selected_brands
            
            # Show a success message with spinner (dummy functionality)
            with st.spinner("Saving your preferences..."):
                # Simulate processing time without actually saving to DB
                time.sleep(1.5)
            
            # Display success popup
            st.success("Your preferences are saved!")
                
            # Show confirmation and continue button
            st.info("You can now proceed to the chat interface to ask about credit card offers.")
            if st.button("Continue to Chat"):
                st.session_state.page = 'chat_interface'
                st.rerun()
        else:
            st.warning("Please select at least one brand to continue.")

def chat_interface_page():
    """Third page with the chat interface using RAG."""
    st.title(f"Credit Card Offers Assistant")
    st.subheader(f"Hello {st.session_state.name}! Ask me about credit card offers")
    
    # Display selected brands
    if st.session_state.selected_brands:
        st.write("Your selected brands:")
        cols = st.columns(4)
        for i, brand in enumerate(st.session_state.selected_brands):
            cols[i % 4].markdown(f"• {brand}")
        st.divider()
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Chat input
    user_query = st.chat_input("Ask about credit card offers...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        st.chat_message("user").write(user_query)
        
        # Display assistant response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Searching for relevant offers..."):
                # Retrieve documents from Astra DB
                results = retrieve_documents(user_query, st.session_state.selected_brands)
                
                if results:
                    # Generate response based on retrieved documents
                    response = generate_answer(user_query, results)
                else:
                    response = "I couldn't find any relevant information based on your query and selected brands."
                
                # Display the response
                st.write(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# Main app logic
def main():
    if st.session_state.page == 'name_input':
        name_input_page()
    elif st.session_state.page == 'brand_selection':
        brand_selection_page()
    elif st.session_state.page == 'chat_interface':
        chat_interface_page()

if __name__ == "__main__":
    main()