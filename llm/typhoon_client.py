"""
Typhoon LLM Client
Handles Thai language conversation through Ollama
"""

import requests
import json
from typing import Dict, Optional, List
from loguru import logger


class TyphoonClient:
    def __init__(self, api_url: str = "http://localhost:11434", 
                 model: str = "typhoon:7b-instruct"):
        """
        Initialize Typhoon LLM client
        
        Args:
            api_url: Ollama API endpoint
            model: Model name in Ollama
        """
        self.api_url = api_url
        self.model = model
        self.generate_endpoint = f"{api_url}/api/generate"
        self.chat_endpoint = f"{api_url}/api/chat"
        
        # Verify connection
        self._verify_connection()
        
    def _verify_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if self.model in model_names:
                    logger.info(f"Connected to Ollama, model '{self.model}' ready")
                else:
                    logger.warning(f"Model '{self.model}' not found. Available: {model_names}")
            else:
                logger.error(f"Failed to connect to Ollama at {self.api_url}")
        except Exception as e:
            logger.error(f"Ollama connection error: {e}")
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 512, stream: bool = False) -> Optional[str]:
        """
        Generate response from prompt
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Generated text or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(self.generate_endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"LLM request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None
    
    def chat(self, messages: List[Dict[str, str]], 
            temperature: float = 0.7, max_tokens: int = 512) -> Optional[str]:
        """
        Multi-turn chat interface
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Assistant response or None
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(self.chat_endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
            else:
                logger.error(f"Chat request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return None
    
    def generate_structured(self, system_prompt: str, user_message: str,
                          temperature: float = 0.5) -> Optional[Dict]:
        """
        Generate structured output (expects JSON response)
        
        Args:
            system_prompt: System instructions
            user_message: User input
            temperature: Lower for more deterministic output
            
        Returns:
            Parsed JSON dict or None
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self.chat(messages, temperature=temperature, max_tokens=512)
        
        if response:
            try:
                # Try to parse JSON
                # Remove markdown code blocks if present
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM: {e}")
                logger.debug(f"Raw response: {response}")
                return None
        
        return None