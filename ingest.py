import argparse
import json
from argparse import RawTextHelpFormatter
import requests
from typing import Optional
import warnings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

# Get configuration from environment variables with fallbacks
BASE_API_URL = os.getenv("LANGFLOW_API_URL", "http://127.0.0.1:7860")
FLOW_ID = os.getenv("LANGFLOW_FLOW_ID", "71625e01-7b72-42aa-a960-8a194f0ab2f3")
ENDPOINT = os.getenv("LANGFLOW_ENDPOINT", "")  # You can set a specific endpoint name in the flow settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
  "FirecrawlScrapeApi-Me0rq": {
    "api_key": FIRECRAWL_API_KEY,
    "timeout": 5,
    "url": ""
  },
  "TextInput-tJ7A3": {
    "input_value": "{\n  \"Restaurants\": [\"thefarmersdog\", \"Chipotle\", \"Olive Garden\", \"Shake Shack\", \"Sweetgreen\"],\n  \"Entertainment\": [\"Netflix\", \"HBO Max\", \"Sony PlayStation\", \"AMC Theatres\", \"Spotify\"],\n  \"Clothing\": [\"Nike\", \"H&M\", \"Zara\", \"Levi's\", \"Adidas\"],\n  \"Travel\": [\"Marriott\", \"Airbnb\", \"Delta Airlines\", \"Expedia\", \"Royal Caribbean\"],\n  \"Health & Fitness\": [\"Peloton\", \"Planet Fitness\", \"Lululemon\", \"Fitbit\", \"Herbalife\"],\n  \"Technology & Gadgets\": [\"Apple\", \"Samsung\", \"Sony\", \"Bose\", \"Microsoft\"],\n  \"Home & Decor\": [\"IKEA\", \"Pottery Barn\", \"Wayfair\", \"West Elm\", \"Crate & Barrel\"],\n  \"Beauty & Personal Care\": [\"Sephora\", \"Glossier\", \"Dove\", \"Olay\", \"L'Oréal\"],\n  \"Books & Literature\": [\"Amazon Books\", \"Barnes & Noble\", \"Audible\", \"Kindle\", \"Penguin Random House\"],\n  \"Outdoor & Adventure\": [\"REI\", \"North Face\", \"Patagonia\", \"Columbia\", \"Yeti\"]\n}"
  },
  "BrandUrlGenerator-8xOBl": {
    "json_input": ""
  },
  "UrlPreprocessor-WWb8Z": {},
  "LoopComponent-CNMbk": {},
  "ParseData-UGQLd": {
    "sep": "\n",
    "template": "{text}"
  },
  "ParseData-aYhw4": {
    "sep": "\n",
    "template": "{markdown}"
  },
  "JSONCleaner-kaDji": {
    "json_str": "",
    "normalize_unicode": False,
    "remove_control_chars": False,
    "validate_json": False
  },
  "Agent-IuFh3": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": OPENAI_API_KEY,
    "handle_parsing_errors": True,
    "input_value": "",
    "json_mode": False,
    "max_iterations": 15,
    "max_retries": 5,
    "max_tokens": None,
    "model_kwargs": {},
    "model_name": "gpt-4o-mini",
    "n_messages": 100,
    "openai_api_base": "",
    "order": "Ascending",
    "seed": 1,
    "sender": "Machine and User",
    "sender_name": "",
    "session_id": "",
    "system_prompt": "Be my text parser and give me a json output of  only the \"showing offers\":\n| Merchant | Offer | Description | Exp | Bank |",
    "temperature": 0.1,
    "template": "{sender_name}: {text}",
    "timeout": 700,
    "verbose": True
  },
  "ParseData-JyDjO": {
    "sep": "\n",
    "template": "{markdown}"
  },
  "StoreMessage-ZXByI": {
    "message": "",
    "sender": "",
    "sender_name": "",
    "session_id": ""
  },
  "MessagetoData-NEiy1": {
    "message": ""
  }
}

def run_flow(message: str,
  endpoint: str,
  output_type: str = "chat",
  input_type: str = "chat",
  tweaks: Optional[dict] = None,
  api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "message": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="""Run a flow with a given message and optional tweaks.
Run it like: python <your file>.py "your message here" --endpoint "your_endpoint" --tweaks '{"key": "value"}'""",
        formatter_class=RawTextHelpFormatter)
    parser.add_argument("message", type=str, help="The message to send to the flow")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT or FLOW_ID, help="The ID or the endpoint name of the flow")
    parser.add_argument("--tweaks", type=str, help="JSON string representing the tweaks to customize the flow", default=json.dumps(TWEAKS))
    parser.add_argument("--api_key", type=str, help="API key for authentication", default=None)
    parser.add_argument("--output_type", type=str, default="chat", help="The output type")
    parser.add_argument("--input_type", type=str, default="chat", help="The input type")
    parser.add_argument("--upload_file", type=str, help="Path to the file to upload", default=None)
    parser.add_argument("--components", type=str, help="Components to upload the file to", default=None)

    args = parser.parse_args()
    try:
      tweaks = json.loads(args.tweaks)
    except json.JSONDecodeError:
      raise ValueError("Invalid tweaks JSON string")

    if args.upload_file:
        if not upload_file:
            raise ImportError("Langflow is not installed. Please install it to use the upload_file function.")
        elif not args.components:
            raise ValueError("You need to provide the components to upload the file to.")
        tweaks = upload_file(file_path=args.upload_file, host=BASE_API_URL, flow_id=args.endpoint, components=[args.components], tweaks=tweaks)

    response = run_flow(
        message=args.message,
        endpoint=args.endpoint,
        output_type=args.output_type,
        input_type=args.input_type,
        tweaks=tweaks,
        api_key=args.api_key
    )

    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()