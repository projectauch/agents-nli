import asyncio
import os
from openai import OpenAI
from typing import List, Dict, Any, Optional

# This should be configured in a more secure way, e.g., environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

async def get_model_response_with_retry(client: OpenAI, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int, call_description: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Generic function to call the OpenAI API with retries."""
    retries = 3
    delay = 30  # seconds
    for attempt in range(retries):
        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if seed is not None:
                params["seed"] = seed

            response = await asyncio.to_thread(
                client.chat.completions.create,
                **params
            )
            return {
                "content": response.choices[0].message.content,
                "success": True
            }
        except Exception as e:
            if attempt < retries - 1:
                print(f"API call failed for '{call_description}' (Attempt {attempt + 1}/{retries}). Retrying in {delay}s... Error: {e}")
                await asyncio.sleep(delay)
            else:
                print(f"API call failed for '{call_description}' on final attempt. Error: {e}")
                return {
                    "content": f"Error after {retries} attempts: {str(e)}",
                    "success": False
                }
    return {"content": "An unexpected error occurred in get_model_response_with_retry", "success": False}
