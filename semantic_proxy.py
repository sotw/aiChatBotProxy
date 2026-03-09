from FlagEmbedding import BGEM3FlagModel
import numpy as np
import re
import requests

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def extract_location(text):
    patterns = [
        r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'at\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

def get_weather(location):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}"
        geo_response = requests.get(geo_url, timeout=30)
        geo_data = geo_response.json()
        
        if 'results' not in geo_data or len(geo_data['results']) == 0:
            return f"Location '{location}' not found"
        
        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        city_name = geo_data['results'][0]['name']
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_response = requests.get(weather_url, timeout=30)
        weather_data = weather_response.json()
        
        current = weather_data['current_weather']
        temp = current['temperature']
        weather_code = current['weathercode']
        
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle",
            53: "Moderate drizzle", 55: "Dense drizzle", 61: "Slight rain",
            63: "Moderate rain", 65: "Heavy rain", 71: "Slight snow",
            73: "Moderate snow", 75: "Heavy snow", 80: "Slight rain showers",
            81: "Moderate rain showers", 82: "Violent rain showers", 95: "Thunderstorm"
        }
        condition = weather_codes.get(weather_code, f"Code: {weather_code}")
        
        return f"Weather in {city_name}: {condition}, Temperature: {temp}°C"
    except Exception as e:
        return f"Error fetching weather: {e}"

ACTIONS = {
    "send_email": "Compose and send an electronic mail to a recipient.",
    "get_weather": "Fetch the current temperature and atmospheric conditions for a city.",
    "query_database": "Search the internal SQL database for customer records.",
    "shutdown_system": "Safely terminate all running processes and power off."
}

action_names = list(ACTIONS.keys())
action_descriptions = list(ACTIONS.values())

# Pre-compute the semantic "fingerprints" of your actions
action_embeddings = model.encode(action_descriptions)['dense_vecs']

def execute_action(action_name, details):
    print(f"--- [EXECUTING ACTION: {action_name}] with context: '{details}' ---")
    if action_name == "get_weather":
        location = extract_location(details)
        if location:
            result = get_weather(location)
            print(f"Result: {result}")
        else:
            print("Result: Could not extract location from input")

def agent_logic(user_input):
    # Step 1: Breakdown/Embed the input
    print(f"Agent is analyzing: '{user_input}'")
    query_emb = model.encode([user_input])['dense_vecs']
    
    # Step 2: Semantic Matching (Cosine Similarity)
    # We find which action description is closest to the user's request
    scores = query_emb @ action_embeddings.T
    best_match_idx = np.argmax(scores)
    confidence = scores[0][best_match_idx]
    
    chosen_action = action_names[best_match_idx]
    
    # Step 3: Threshold check (Don't execute if it's a weak match)
    if confidence > 0.45: 
        execute_action(chosen_action, user_input)
    else:
        print("Agent Logic: No clear action found. Responding with general chat.")

# --- Interactive CLI ---
if __name__ == "__main__":
    print("=== Semantic Agent CLI ===")
    print("Available actions: send_email, get_weather, query_database, shutdown_system")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break
        if not user_input:
            continue
        agent_logic(user_input)
        print()
