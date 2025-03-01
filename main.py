import streamlit as st
import os
import time
import json
import pandas as pd
import requests
from openai import OpenAI
from astrapy import DataAPIClient
from dotenv import load_dotenv
from streamlit_js_eval import get_geolocation

# Load environment variables from .env file
load_dotenv()

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
if 'top_restaurant' not in st.session_state:
    st.session_state.top_restaurant = None
if 'current_offer' not in st.session_state:
    st.session_state.current_offer = None

# Define the brands extracted from your data
brands = [
    "Mc Donald's",
    "Taco Bell",
    "Acropolispizzapasta",
    "Acropolispizzapastaeverett",
    "Starbucks",
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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set it as an environment variable in the .env file.")
        return None
    return OpenAI(api_key=api_key)

def get_astra_collection():
    """Connect to Astra DB and return the collection."""
    try:
        app_token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
        api_endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT")
        
        if not app_token or not api_endpoint:
            st.error("Astra DB credentials not found. Please set them as environment variables in the .env file.")
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

# Restaurant finder and notification functions

def find_top_restaurant(latitude, longitude, radius=1000):
    """Find the top restaurant near the given coordinates"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Overpass query to find restaurants
    query = f"""
    [out:json];
    (
      node["amenity"="restaurant"](around:{radius},{latitude},{longitude});
      way["amenity"="restaurant"](around:{radius},{latitude},{longitude});
      relation["amenity"="restaurant"](around:{radius},{latitude},{longitude});
    );
    out center;
    """
    
    try:
        response = requests.post(overpass_url, data=query)
        data = response.json()
        elements = data.get("elements", [])
        
        if not elements:
            return None
        
        # Get the first (closest) restaurant
        restaurant = elements[0]
        tags = restaurant.get("tags", {})
        
        return {
            "name": tags.get("name", "Unnamed Restaurant"),
            "street": tags.get("addr:street", ""),
            "housenumber": tags.get("addr:housenumber", ""),
            "city": tags.get("addr:city", ""),
            "lat": restaurant.get("lat") or restaurant.get("center", {}).get("lat"),
            "lon": restaurant.get("lon") or restaurant.get("center", {}).get("lon"),
        }
        
    except Exception as e:
        st.error(f"Error finding restaurants: {e}")
        return None

def find_offers_for_restaurant(restaurant_name):
    """Use the RAG system to find offers for the given restaurant"""
    try:
        # Use the existing OpenAI client
        client = get_openai_client()
        if not client:
            st.error("OpenAI API client not available.")
            return None
        
        # Generate a query to find offers for this restaurant
        query = f"What are the best credit card offers or rewards for {restaurant_name}?"
        
        # Generate embedding for the query
        embedding_response = client.embeddings.create(
            input=query, 
            model="text-embedding-3-small"
        )
        query_embedding = embedding_response.data[0].embedding
        
        # Get Astra collection
        collection = get_astra_collection()
        if not collection:
            st.error("Could not connect to Astra DB.")
            return None
        
        # Search for relevant documents
        results = collection.find(
            sort={"$vector": query_embedding},
            limit=3
        )
        
        if not results:
            # If no specific results, create a generic offer
            return {
                "title": f"Special Offer for {restaurant_name}",
                "value": "Use your credit card to earn 2x points on dining"
            }
        
        # Combine context from results
        context = "\n\n".join([doc.get("content", "") for doc in results])
        
        # Create prompt to extract the best offer
        prompt = f"""
        You are an AI assistant helping users find credit card offers and rewards.
        Based on the following information, identify the BEST credit card offer or reward for {restaurant_name}.
        If there's no specific offer for this restaurant, recommend a general dining/restaurant reward.

        Here's the information:
        {context}

        Provide your response in JSON format with these fields:
        1. "title": A catchy, short title for the offer (max 50 chars)
        2. "value": A brief description of the value (max 100 chars)

        JSON response only, no additional text.
        """
        
        # Generate response
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in credit card offers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        
        response_text = completion.choices[0].message.content
        
        # Parse JSON from response
        # First, find JSON object in the response if there's additional text
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                offer_data = json.loads(json_str)
            else:
                offer_data = json.loads(response_text)
                
            return offer_data
            
        except json.JSONDecodeError:
            # If we can't parse JSON, create a simple offer from the response
            return {
                "title": f"Offer for {restaurant_name}",
                "value": response_text[:100] if len(response_text) > 100 else response_text
            }
            
    except Exception as e:
        st.error(f"Error finding offers: {e}")
        return None

def send_notification(restaurant_name, offer_title, offer_value):
    """Send a notification using the Langflow API"""
    try:
        # Get API credentials from environment variables
        langflow_api_url = os.environ.get("LANGFLOW_API_URL")
        api_token = os.environ.get("LANGFLOW_API_TOKEN")
        
        if not langflow_api_url or not api_token:
            st.error("Langflow API credentials not configured.")
            return False
        
        # Default URL if not provided in environment
        if not langflow_api_url:
            langflow_api_url = "https://api.langflow.astra.datastax.com/lf/9b25fb9f-2ae3-46fa-859b-91d46528ff89/api/v1/run/62bf4e2d-fee9-4c64-8c0f-141a27eb5de2?stream=false"
        
        # Prepare the payload
        payload = {
            "input_value": f"New offer at {restaurant_name}: {offer_title}",
            "output_type": "chat",
            "input_type": "chat",
            "tweaks": {
                "TwilioFlowTrigger-xkhsT": {
                    "contact_address": "+16099178654",
                    "offer_title": offer_title,
                    "offer_value": offer_value,
                    "twilio_account_sid": os.environ.get("TWILIO_ACCOUNT_SID", "AC22f5162761778232fdfcdefac4d26f63"),
                    "twilio_auth_token": os.environ.get("TWILIO_AUTH_TOKEN", "6bd348defe8030f5a41813830ec46b28"),
                    "twilio_phone_number": os.environ.get("TWILIO_PHONE_NUMBER", "+18669742432")
                }
            }
        }
        
        # Make the API request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_token}'
        }
        
        response = requests.post(langflow_api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        st.error(f"Error sending notification: {e}")
        return False

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
    """Second page to select brand preferences and chat interface."""
    st.title(f"Hi {st.session_state.name}, Welcome to Credit Card Offers Assistant")
    
    # Create tabs for different features
    tab1, tab2 = st.tabs(["Chat Assistant", "Restaurant Finder"])
    
    with tab1:
        # Original brand selection and chat interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Choose Your Preferred Brands")
            
            selected_brands = st.multiselect(
                "Select brands you prefer:",
                options=brands,
                default=st.session_state.selected_brands
            )
            
            if st.button("Save Preferences"):
                if selected_brands:
                    st.session_state.selected_brands = selected_brands
                    
                    with st.spinner("Saving your preferences..."):
                        time.sleep(1.5)
                    
                    st.success("Your preferences are saved!")
                else:
                    st.warning("Please select at least one brand to continue.")
            
            # Display selected brands
            if st.session_state.selected_brands:
                st.write("Your selected brands:")
                for brand in st.session_state.selected_brands:
                    st.write(f"• {brand}")
        
        with col2:
            st.subheader("Ask About Credit Card Offers")
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
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
                with chat_container:
                    st.chat_message("user").write(user_query)
                
                # Display assistant response with spinner
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Searching for relevant offers..."):
                            # Generate embedding for query
                            client = get_openai_client()
                            if not client:
                                st.error("OpenAI API key not found or invalid.")
                                response = "I'm having trouble connecting to my knowledge base. Please check the API configuration."
                            else:
                                try:
                                    # Generate embedding
                                    embedding_response = client.embeddings.create(
                                        input=user_query, 
                                        model="text-embedding-3-small"
                                    )
                                    query_embedding = embedding_response.data[0].embedding
                                    
                                    # Connect to Astra DB
                                    collection = get_astra_collection()
                                    if not collection:
                                        st.error("Could not connect to Astra DB.")
                                        response = "I'm having trouble connecting to my knowledge base. Please check the database configuration."
                                    else:
                                        # Perform vector search
                                        results = collection.find(
                                            sort={"$vector": query_embedding},
                                            limit=5
                                        )
                                        
                                        if not results:
                                            response = "I couldn't find any relevant information for your question in my database."
                                        else:
                                            # Combine document content to create context
                                            context = "\n\n".join([doc.get("content", "") for doc in results])
                                            
                                            # Create prompt with the retrieved context
                                            prompt = f"""
                                            You are an AI assistant helping users find information about credit card offers and rewards.
                                            Answer the user's question using ONLY the following information from the database:

                                            {context}

                                            If the information needed to answer the question is not in the provided context, say "I don't have that information in my database."
                                            Be concise and helpful. Format any offers nicely.

                                            User question: {user_query}
                                            """
                                            
                                            # Generate response
                                            completion = client.chat.completions.create(
                                                model="gpt-4o",
                                                messages=[
                                                    {"role": "system", "content": "You are a helpful assistant specialized in credit card offers."},
                                                    {"role": "user", "content": prompt}
                                                ],
                                                temperature=0.3,
                                            )
                                            
                                            response = completion.choices[0].message.content
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    response = "I encountered an error while processing your request. Please try again later."
                            
                            # Display the response
                            st.write(response)
                            
                            # Add assistant response to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with tab2:
        # Restaurant finder feature
        st.subheader("Find Restaurant Offers Near You")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            find_restaurant = st.button("Find Nearest Restaurant")
            
            if find_restaurant:
                with st.spinner("Detecting your location..."):
                    loc = get_geolocation()
                    
                    if not loc or 'coords' not in loc:
                        st.error("Could not detect your location. Please allow location access.")
                    else:
                        # Get coordinates
                        latitude = loc.get('coords', {}).get('latitude')
                        longitude = loc.get('coords', {}).get('longitude')
                        
                        st.success(f"Located you at: {latitude}, {longitude}")
                        
                        # Find the top restaurant near the user
                        top_restaurant = find_top_restaurant(latitude, longitude)
                        
                        if not top_restaurant:
                            st.warning("No restaurants found near your location.")
                        else:
                            # Display the restaurant
                            restaurant_name = top_restaurant.get("name", "Unnamed Restaurant")
                            st.session_state.top_restaurant = restaurant_name
                            
                            # Get address
                            street = top_restaurant.get("street", "")
                            housenumber = top_restaurant.get("housenumber", "")
                            city = top_restaurant.get("city", "")
                            address = ", ".join(filter(None, [f"{housenumber} {street}".strip(), city])) or "No address provided"
                            
                            st.markdown(f"### {restaurant_name}")
                            st.write(f"Address: {address}")
                            
                            # Find offers for this restaurant
                            st.info("Searching for offers...")
                            offer = find_offers_for_restaurant(restaurant_name)
                            
                            if offer:
                                st.session_state.current_offer = offer
                                offer_title = offer.get("title", "Special Offer")
                                offer_value = offer.get("value", "Discount available")
                                
                                st.success(f"Found an offer: {offer_title}")
                                st.markdown(f"**{offer_title}**")
                                st.markdown(f"{offer_value}")
                                
                                # Show notification button
                                if st.button("Send this offer to my phone"):
                                    send_notification(restaurant_name, offer_title, offer_value)
                                    st.success("Notification sent to your phone!")
                            else:
                                st.warning("No special offers found for this restaurant.")
        
        with col2:
            if 'top_restaurant' in st.session_state and st.session_state.top_restaurant and 'current_offer' in st.session_state and st.session_state.current_offer:
                # Display card for the current offer
                st.markdown("### Current Offer")
                offer_card = f"""
                <div style="border:1px solid #ddd; padding:10px; border-radius:5px;">
                    <h4>{st.session_state.top_restaurant}</h4>
                    <h5>{st.session_state.current_offer.get('title', 'Special Offer')}</h5>
                    <p>{st.session_state.current_offer.get('value', 'Discount available')}</p>
                </div>
                """
                st.markdown(offer_card, unsafe_allow_html=True)

# Main app logic
def main():
    if st.session_state.page == 'name_input':
        name_input_page()
    elif st.session_state.page == 'brand_selection':
        brand_selection_page()

if __name__ == "__main__":
    main()