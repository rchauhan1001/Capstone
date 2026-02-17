# Configuration settings for Metamind system

# LLM API settings
LLM_CONFIG = {
    "api_key": "your-openai-key",
    "base_url": "your-openai-url",
    "model_name": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000
}

# ToM Agent settings
TOM_AGENT_CONFIG = {
    "hypothesis_count": 7, 
    "target_diversity": 0.4,  
    "evidence_threshold": "medium-high"  
}

# Domain Agent settings
DOMAIN_AGENT_CONFIG = {
    "lambda": 0.7, 
    "epsilon": 1e-10  # Small constant to avoid log(0)
}

# Response Agent settings
RESPONSE_AGENT_CONFIG = {
    "beta": 0.8,  # Trade-off weight for empathy vs coherence
    "utility_threshold": 0.9,  # Threshold for acceptable utility score
    "max_revisions": 3  # Maximum number of response revisions
}

# Social Memory settings
SOCIAL_MEMORY_CONFIG = {
    "memory_decay_rate": 0.05,  # Rate at which memory importance decays over time
    "max_memory_items": 100  # Maximum number of items to store in memory
}

# Mental state types
MENTAL_STATE_TYPES = ["Belief", "Desire", "Intention", "Emotion", "Thought"]