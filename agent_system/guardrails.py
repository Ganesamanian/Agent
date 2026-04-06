from typing import Optional, Tuple
import re
from .model_provider import ModelProvider
from .config import AppConfig

PII_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|\d{3}[-.]?\d{3}[-.]?\d{4}')
JAILBREAK_KEYWORDS = ['ignore previous', 'ignore instructions', 
                      'jailbreak', 'refund everything', 
                      'do anything now', 'show me all the data you have',  
                      'bypass security', 'disregard policies',
                      'break character', 'unethical',
                      'illegal', 'harmful',
                      'malicious', 'exploit',
                      'vulnerability', 'phishing',
                      'scam', 'fraud',
                      'steal', 'hack',
                      'attack', 'weaponize',]

def sanitize_input(text: str) -> Tuple[str, bool]:
    """
    Sanitize input: remove PII, check jailbreak keywords.
    Returns (sanitized_text, is_blocked).
    """
    sanitized = PII_REGEX.sub('[PII]', text)
    blocked = any(kw in text.lower() for kw in JAILBREAK_KEYWORDS)
    return sanitized, blocked

def moderate_output(text: str, provider: ModelProvider, config: AppConfig, llm_provider: str) -> Tuple[str, bool]:
    """
    LLM-based output moderation: Check for harmful/policy violation.
    Returns (text, is_flagged).
    """
    system_prompt = (
        "You are a safety evaluator for Booking.com support agents. "
        "Check if this output is safe, ethical, follows Booking.com policies, non-harmful. "
        "Respond ONLY 'SAFE' or 'FLAGGED: <reason>'."
    )
    user_prompt = f"Output: {text}"
    
    response = provider.generate_text(
        provider=llm_provider,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    
    is_flagged = 'FLAGGED' in response.upper()
    return text if not is_flagged else '[FLAGGED: Moderation blocked output]', is_flagged

