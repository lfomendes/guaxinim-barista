"""
Cache Module
Provides persistent caching functionality using Redis.
"""

from dataclasses import asdict, is_dataclass
from functools import wraps
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable
import redis
from guaxinim.core.logger import logger

# Initialize Redis client
def clear_cache() -> None:
    """Clear all cached responses from Redis."""
    try:
        # Delete all keys with our prefix
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(cursor, match="guaxinim:*")
            if keys:
                redis_client.delete(*keys)
            if cursor == 0:
                break
        logger.info("🧹 Cache cleared successfully!")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

def persistent_cache(ttl_days: int = 30) -> Callable:
    """
    A decorator that provides persistent caching using Redis.
    
    Args:
        ttl_days (int): Number of days to keep the cache valid
        
    Returns:
        Callable: The decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create a unique key based on function name and arguments
            # Get the argument names from the function's signature
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # Create args_dict excluding 'self'
            if param_names and param_names[0] == 'self':
                args_dict = {}
                if len(args) > 1:
                    for i, arg in enumerate(args[1:]):
                        # Convert dataclass to dict if it is one
                        if is_dataclass(arg):
                            args_dict[param_names[i+1]] = asdict(arg)
                        else:
                            args_dict[param_names[i+1]] = arg
            else:
                args_dict = {}
                for i, arg in enumerate(args):
                    # Convert dataclass to dict if it is one
                    if is_dataclass(arg):
                        args_dict[param_names[i]] = asdict(arg)
                    else:
                        args_dict[param_names[i]] = arg
                
            # Get instance attributes if this is a method call
            instance_attrs = {}
            if args and hasattr(args[0], 'similarity_field'):
                instance_attrs['similarity_field'] = args[0].similarity_field
            
            # Create the cache key
            key_dict = {
                'func_name': func.__name__,
                'args': args_dict,
                'kwargs': kwargs,
                'instance_attrs': instance_attrs  # Include relevant instance attributes
            }
            cache_key = f"guaxinim:{json.dumps(key_dict, sort_keys=True)}"
            
            # Try to get from cache first
            try:
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    logger.info(f"✨ Using cached response for {func.__name__} - Saved an API call!")
                    return result
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
            
            # Execute the function if not in cache
            result = func(*args, **kwargs)
            
            # Save to cache
            try:
                pickled_result = pickle.dumps(result)
                redis_client.setex(
                    cache_key,
                    ttl_days * 24 * 60 * 60,  # TTL in seconds
                    pickled_result
                )
                logger.info(f"💾 Caching response for {func.__name__} for future use")
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
            
            return result
        return wrapper
    return decorator
