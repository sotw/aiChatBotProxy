#!/usr/bin/env python3
import argparse
import requests
import sys

def main():
    parser = argparse.ArgumentParser(description="Semantic Proxy Client")
    parser.add_argument('query', type=str, help='Query text to send')
    parser.add_argument('--url', default='http://127.0.0.1:8080/predict', help='Server URL')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Server port')
    parser.add_argument('--silence', action='store_true', help='Silence mode')
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/predict"
    
    try:
        response = requests.post(url, json={'text': args.query}, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if args.silence:
            if result.get('result'):
                print(result.get('result'))
            else:
                print(result.get('action', ''))
        else:
            print(f"Action: {result.get('action')}")
            print(f"Confidence: {result.get('confidence'):.4f}")
            if result.get('result'):
                print(f"Result: {result.get('result')}")
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Is it running?", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
