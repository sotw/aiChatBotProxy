import os
import sys
import io
import warnings
from contextlib import redirect_stdout, redirect_stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

from FlagEmbedding import BGEM3FlagModel
import argparse
import numpy as np
import re
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import json
from deep_translator import GoogleTranslator

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")

model = None
action_embeddings = None
translator_cache = {}

def detect_language(text):
    ja_hiragana = re.findall(r'[\u3040-\u309f]', text)
    ja_katakana = re.findall(r'[\u30a0-\u30ff]', text)
    if ja_hiragana or ja_katakana:
        return 'ja'
    
    if re.search(r'[\u4e00-\u9fff]', text):
        common_trad = set('為說時國這個沒會對於發後來過嗎與魚東齊寶車馬門見幾風鳥龍龜壓聲興關臺灣復攜')
        common_simplified = set('为说时国这个没会对于发后来过吗与鱼东齐宝车马门见几风鸟龙龟压声兴关台湾复携')
        
        text_set = set(text)
        if text_set & common_trad and not (text_set & common_simplified - common_trad):
            return 'zh-TW'
        if text_set & common_simplified:
            return 'zh-CN'
        return 'zh-CN'
    
    return 'en'

def translate_to_english(text, src_lang=None):
    if src_lang is None:
        src_lang = detect_language(text)
    if src_lang == 'en':
        return text, 'en'
    lang_code = src_lang
    if src_lang in ('zh', 'zh-CN'):
        lang_code = 'zh-CN'
    elif src_lang == 'zh-TW':
        lang_code = 'zh-TW'
    cache_key = (text, lang_code, 'en')
    if cache_key in translator_cache:
        return translator_cache[cache_key], src_lang
    try:
        result = GoogleTranslator(source=lang_code, target='en').translate(text)
        translator_cache[cache_key] = result
        return result, src_lang
    except:
        return text, 'en'

def translate_from_english(text, target_lang):
    if target_lang == 'en':
        return text
    lang_code = target_lang
    if target_lang in ('zh', 'zh-CN'):
        lang_code = 'zh-CN'
    elif target_lang == 'zh-TW':
        lang_code = 'zh-TW'
    elif target_lang == 'ja':
        lang_code = 'ja'
    cache_key = (text, 'en', lang_code)
    if cache_key in translator_cache:
        return translator_cache[cache_key]
    try:
        result = GoogleTranslator(source='en', target=lang_code).translate(text)
        translator_cache[cache_key] = result
        return result
    except:
        return text

def _get_model():
    print(f"Loading model from: {MODEL_PATH}")
    return BGEM3FlagModel(MODEL_PATH, use_fp16=True)

def init_model(silence):
    global model, action_embeddings
    devnull = io.StringIO()
    with redirect_stdout(devnull), redirect_stderr(devnull):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _get_model()
            action_embeddings = model.encode(list(ACTIONS.values()))['dense_vecs']

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
    
    ja_location_patterns = [
        r'東京|大阪|京都|名古屋|札幌|福岡|横浜|川崎|神戸|広島|仙台|新潟|熊本|鹿児島|沖縄|長野|山形|北海道|福岡',
    ]
    for pattern in ja_location_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
            
    zh_location_patterns = [
        r'北京|上海|广州|深圳|香港|台北|天津|重庆|成都|杭州|南京|武汉|西安|苏州|青岛|厦门|长沙|郑州|沈阳|大连',
    ]
    for pattern in zh_location_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    
    return None

def extract_action_from_input(text):
    text_lower = text.lower()
    
    stopwords = {"a", "an", "the", "in", "at", "to", "for", "of", "you", "me", "my", "i", "your", "is", "are", "was", "will", "could", "would", "should", "can", "do", "does", "want", "like"}
    
    action_keywords = [
        ("get me a", "get"), ("get me", "get"), ("get a", "get"), ("get the", "get"), ("get ", "get"),
        ("fetch ", "fetch"), ("bring me", "bring"), ("bring a", "bring"), ("bring ", "bring"),
        ("make ", "make"), ("order ", "order"), ("buy ", "buy"),
        ("send ", "send"), ("email ", "email"), ("mail ", "mail"),
        ("check ", "check"), ("look up ", "lookup"), ("search ", "search"), ("find ", "find"), ("know ", "know"),
        ("what is ", "query"), ("what's ", "query"), ("tell me ", "query"),
        ("weather", "weather"), ("temperature", "weather"),
        ("shutdown", "shutdown"), ("turn off ", "shutdown"), ("power off ", "shutdown"),
        ("delete ", "delete"), ("remove ", "remove"),
        ("create ", "create"), ("add ", "add"),
        ("update ", "update"), ("edit ", "edit"), ("modify ", "modify"),
    ]
    
    for phrase, action in action_keywords:
        if phrase in text_lower:
            idx = text_lower.find(phrase) + len(phrase)
            remaining = text_lower[idx:].strip()
            remaining_words = remaining.split()
            target_words = []
            for w in remaining_words:
                if w in stopwords:
                    continue
                target_words.append(w)
                if len(target_words) >= 2:
                    break
            target = " ".join(target_words) if target_words else None
            return action, target
    
    words = re.findall(r'[a-z]+', text_lower)
    if words:
        filtered = [w for w in words if w not in stopwords]
        return filtered[0] if filtered else None, filtered[1] if len(filtered) > 1 else None
    return None, None

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
    "shutdown_system": "Safely terminate all running processes and power off.",
    "general_chat": "Handle general conversation, greetings, and non-action queries."
}

action_names = list(ACTIONS.keys())

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/actions':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "actions": ACTIONS,
                "model": "BAAI/bge-m3"
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            result = agent_logic(data.get('text', ''), return_result=True, execute_action_flag=True)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass

def run_server(port):
    server = HTTPServer(('127.0.0.1', port), RequestHandler)
    print(f"Server running on http://127.0.0.1:{port}")
    print(f"POST /predict with {{'text': 'your input'}}")
    server.serve_forever()

def silent_encode(texts):
    devnull = io.StringIO()
    with redirect_stdout(devnull), redirect_stderr(devnull):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.encode(texts)['dense_vecs']

def log(msg):
    if not args.silence:
        print(msg)

def execute_action(action_name, original_text, translated_text):
    log(f"--- [EXECUTING ACTION: {action_name}] with context: '{original_text}' ---")
    if action_name == "general_chat":
        msg = "I'm here to help with specific actions like weather, email, database queries, etc. How can I assist you?"
        print(msg if args.silence else f"Result: {msg}")
        return
    if action_name == "get_weather":
        location = extract_location(original_text) or extract_location(translated_text)
        if location:
            if detect_language(location) != 'en':
                location_en, _ = translate_to_english(location, detect_language(location))
                location = location_en
            result = get_weather(location)
            print(result if args.silence else f"Result: {result}")
        else:
            print("Could not extract location from input" if args.silence else "Result: Could not extract location from input")

def agent_logic(user_input, return_result=False, execute_action_flag=True):
    src_lang = detect_language(user_input)
    
    query_emb_direct = silent_encode([user_input])
    scores_direct = query_emb_direct @ action_embeddings.T
    
    all_actions_direct = [(action_names[i], float(scores_direct[0][i])) for i in range(len(action_names))]
    all_actions_direct.sort(key=lambda x: x[1], reverse=True)
    
    best_match_idx_direct = np.argmax(scores_direct)
    confidence_direct = scores_direct[0][best_match_idx_direct]
    chosen_action_direct = action_names[best_match_idx_direct]
    location_direct = extract_location(user_input)
    
    extracted_action, extracted_target = extract_action_from_input(user_input)
    log(f"Agent analyzing (direct): '{user_input}' (detected: {src_lang})")
    log(f"  -> Extracted action: {extracted_action}, target: {extracted_target}")
    log(f"  -> Possible actions (ranked): {all_actions_direct}")
    log(f"  -> Best action: {chosen_action_direct} (conf: {confidence_direct:.3f})")
    log(f"  -> Extracted params: {json.dumps({'location': location_direct})}")
    
    if confidence_direct > 0.35:
        chosen_action = action_names[best_match_idx_direct]
        
        if chosen_action == "get_weather" and not (extract_location(user_input) or extract_location(user_input)):
            chosen_action = "general_chat"
        
        result = None
        if execute_action_flag:
            result = execute_action_and_get_result(chosen_action, user_input, user_input)
            if result and src_lang != 'en':
                if src_lang == 'zh-CN':
                    result = translate_from_english(result, 'zh-TW')
                else:
                    result = translate_from_english(result, src_lang)
        
        if return_result:
            return {"action": chosen_action, "confidence": float(confidence_direct), "result": result, "detected_lang": src_lang, "mode": "direct"}
        
        execute_action(chosen_action, user_input, user_input)
        return
    
    en_text, _ = translate_to_english(user_input, src_lang)
    log(f"Agent fallback to translated: '{user_input}' -> '{en_text}'")
    query_emb = silent_encode([en_text])
    
    scores = query_emb @ action_embeddings.T
    
    all_actions = [(action_names[i], float(scores[0][i])) for i in range(len(action_names))]
    all_actions.sort(key=lambda x: x[1], reverse=True)
    
    best_match_idx = np.argmax(scores)
    confidence = scores[0][best_match_idx]
    
    chosen_action = action_names[best_match_idx]
    location_translated = extract_location(en_text)
    
    extracted_action_en, extracted_target_en = extract_action_from_input(en_text)
    log(f"Agent fallback to translated: '{user_input}' -> '{en_text}'")
    log(f"  -> Extracted action: {extracted_action_en}, target: {extracted_target_en}")
    log(f"  -> Possible actions (ranked): {all_actions}")
    log(f"  -> Best action: {chosen_action} (conf: {confidence:.3f})")
    log(f"  -> Extracted params: {json.dumps({'location': location_translated})}")
    
    if chosen_action == "get_weather" and not (extract_location(user_input) or extract_location(en_text)):
        chosen_action = "general_chat"
    
    result = None
    if execute_action_flag and confidence > 0.45:
        result = execute_action_and_get_result(chosen_action, user_input, en_text)
        if result and src_lang != 'en':
            if src_lang == 'zh-CN':
                result = translate_from_english(result, 'zh-TW')
            else:
                result = translate_from_english(result, src_lang)
    
    if return_result:
        return {"action": chosen_action, "confidence": float(confidence), "result": result, "detected_lang": src_lang, "mode": "translated"}
    
    if confidence > 0.45: 
        execute_action(chosen_action, user_input, en_text)
    else:
        chosen_action = "general_chat"
        log("Agent Logic: No clear action found. Responding with general chat.")

def execute_action_and_get_result(action_name, original_text, translated_text):
    if action_name == "general_chat":
        return "I'm here to help with specific actions like weather, How can I assist you?"
    if action_name == "get_weather":
        location = extract_location(original_text) or extract_location(translated_text)
        if location:
            if detect_language(location) != 'en':
                location_en, _ = translate_to_english(location, detect_language(location))
                location = location_en
            return get_weather(location)
        return "Could not extract location from input"
    return None

def main():
    global args
    parser = argparse.ArgumentParser(description="Semantic Proxy CLI")
    parser.add_argument('-q', '--question', type=str, help='Question to ask (non-interactive mode)')
    parser.add_argument('--silence', action='store_true', help='Clean output without debug logs')
    parser.add_argument('--server', type=int, metavar='PORT', help='Run as HTTP server on specified port')
    args = parser.parse_args()
    
    init_model(args.silence if hasattr(args, 'silence') and args.silence else False)
    
    if args.server:
        run_server(args.server)
    
    if args.question:
        agent_logic(args.question)
    else:
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

if __name__ == "__main__":
    main()
