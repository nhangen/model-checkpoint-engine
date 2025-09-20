"""Intelligent caching system for checkpoint and experiment data"""

import time
import threading
from typing import Dict, Any, Optional, Union, Callable, Tuple, List
from collections import OrderedDict
import pickle
import hashlib


class LRUCache:
    """Thread-safe LRU cache with TTL support"""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        """
        Initialize LRU cache

        Args:
            max_size: Maximum number of items to cache
            default_ttl: Default time-to-live in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._ttls = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache - optimized single-pass implementation"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default

            # Optimized: check expiration and update in single operation
            if self._is_expired(key):
                self._remove_key(key)
                self._misses += 1
                return default

            # Optimized: move to end and return in single operation
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set item in cache - optimized bulk operations"""
        with self._lock:
            current_time = time.time()

            # Optimized: set all values at once
            self._cache[key] = value
            self._timestamps[key] = current_time

            # Optimized: conditional TTL setting
            ttl_value = ttl if ttl is not None else self.default_ttl
            if ttl_value is not None:
                self._ttls[key] = ttl_value

            self._cache.move_to_end(key)

            # Optimized: batch eviction if over capacity
            if len(self._cache) > self.max_size:
                excess = len(self._cache) - self.max_size
                for _ in range(excess):
                    oldest_key = next(iter(self._cache))
                    self._remove_key(oldest_key)

    def delete(self, key: str) -> bool:
        """
        Delete item from cache

        Args:
            key: Cache key

        Returns:
            True if key existed and was deleted
        """
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all items from cache"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttls.clear()
            self._hits = 0
            self._misses = 0

    def _is_expired(self, key: str) -> bool:
        """Check if a key has expired"""
        if key not in self._ttls:
            return False

        ttl = self._ttls[key]
        timestamp = self._timestamps[key]
        return time.time() - timestamp > ttl

    def _remove_key(self, key: str) -> None:
        """Remove key from all internal structures"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._ttls.pop(key, None)

    def _cleanup_expired(self) -> None:
        """Remove all expired items"""
        expired_keys = []
        current_time = time.time()

        for key in list(self._cache.keys()):
            if key in self._ttls:
                ttl = self._ttls[key]
                timestamp = self._timestamps[key]
                if current_time - timestamp > ttl:
                    expired_keys.append(key)

        for key in expired_keys:
            self._remove_key(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate_percent': hit_rate,
                'total_requests': total_requests
            }

    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)


class CheckpointCache:
    """Specialized cache for checkpoint metadata and lightweight data - optimized"""

    def __init__(self, max_size: int = 500, metadata_ttl: float = 3600, data_ttl: float = 1800):
        """Initialize optimized checkpoint cache with efficient key prefixes"""
        # Optimized: pre-allocate caches with efficient sizing
        self.metadata_cache = LRUCache(max_size >> 1, metadata_ttl)  # Bit shift instead of division
        self.data_cache = LRUCache(max_size >> 1, data_ttl)
        self.query_cache = LRUCache(max_size >> 2, metadata_ttl)

        # Optimized: pre-computed key prefixes to avoid string concatenation
        self._metadata_prefix = "m:"
        self._data_prefix = "d:"

    def get_checkpoint_metadata(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get cached checkpoint metadata - optimized key formatting"""
        return self.metadata_cache.get(self._metadata_prefix + checkpoint_id)

    def set_checkpoint_metadata(self, checkpoint_id: str, metadata: Dict[str, Any]) -> None:
        """Cache checkpoint metadata - optimized key formatting"""
        self.metadata_cache.set(self._metadata_prefix + checkpoint_id, metadata)

    def get_checkpoint_data(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get cached checkpoint data - optimized key formatting"""
        return self.data_cache.get(self._data_prefix + checkpoint_id)

    def set_checkpoint_data(self, checkpoint_id: str, data: Dict[str, Any],
                          max_size_mb: float = 50) -> bool:
        """Cache checkpoint data if small enough - optimized size estimation"""
        try:
            # Optimized: estimate size without full serialization for large objects
            import sys
            estimated_size = sys.getsizeof(data)

            # Quick check: if estimated size is too large, skip expensive pickle
            if estimated_size > max_size_mb * 1048576:  # Pre-calculated bytes
                return False

            # Only serialize if initial estimate looks promising
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            if len(serialized) <= max_size_mb * 1048576:
                self.data_cache.set(self._data_prefix + checkpoint_id, data)
                return True
            return False

        except Exception:
            return False

    def get_query_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result"""
        return self.query_cache.get(query_hash)

    def set_query_result(self, query_hash: str, result: Any) -> None:
        """Cache query result"""
        self.query_cache.set(query_hash, result)

    def create_query_hash(self, query_params: Dict[str, Any]) -> str:
        """Create a hash for query parameters"""
        # Sort parameters for consistent hashing
        sorted_params = sorted(query_params.items())
        param_str = str(sorted_params)
        return hashlib.md5(param_str.encode()).hexdigest()

    def invalidate_checkpoint(self, checkpoint_id: str) -> None:
        """Invalidate all cached data for a checkpoint"""
        self.metadata_cache.delete(f"metadata:{checkpoint_id}")
        self.data_cache.delete(f"data:{checkpoint_id}")

    def invalidate_experiment(self, experiment_id: str) -> None:
        """Invalidate all cached data related to an experiment"""
        # This is a simple implementation - in practice, you might want
        # to track experiment->checkpoint relationships for efficient invalidation
        self.query_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'metadata_cache': self.metadata_cache.get_stats(),
            'data_cache': self.data_cache.get_stats(),
            'query_cache': self.query_cache.get_stats(),
            'total_items': (self.metadata_cache.size() +
                          self.data_cache.size() +
                          self.query_cache.size())
        }


class ExperimentCache:
    """Specialized cache for experiment tracking data"""

    def __init__(self, max_size: int = 1000, default_ttl: float = 1800):
        """
        Initialize experiment cache

        Args:
            max_size: Maximum number of items to cache
            default_ttl: Default TTL in seconds (30 minutes)
        """
        self.cache = LRUCache(max_size, default_ttl)

    def get_experiment_metadata(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get cached experiment metadata"""
        return self.cache.get(f"exp_meta:{experiment_id}")

    def set_experiment_metadata(self, experiment_id: str, metadata: Dict[str, Any]) -> None:
        """Cache experiment metadata"""
        self.cache.set(f"exp_meta:{experiment_id}", metadata)

    def get_metrics_data(self, experiment_id: str, metric_filter: str = "") -> Optional[List[Dict]]:
        """Get cached metrics data"""
        cache_key = f"metrics:{experiment_id}:{metric_filter}"
        return self.cache.get(cache_key)

    def set_metrics_data(self, experiment_id: str, metrics: List[Dict],
                        metric_filter: str = "") -> None:
        """Cache metrics data"""
        cache_key = f"metrics:{experiment_id}:{metric_filter}"
        self.cache.set(cache_key, metrics)

    def get_statistics(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get cached experiment statistics"""
        return self.cache.get(f"stats:{experiment_id}")

    def set_statistics(self, experiment_id: str, stats: Dict[str, Any]) -> None:
        """Cache experiment statistics"""
        self.cache.set(f"stats:{experiment_id}", stats)

    def invalidate_experiment(self, experiment_id: str) -> None:
        """Invalidate all cached data for an experiment"""
        # Remove all keys that start with this experiment ID
        keys_to_remove = []
        for key in self.cache._cache.keys():
            if key.endswith(experiment_id) or f":{experiment_id}:" in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self.cache.delete(key)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()


class CacheManager:
    """Central cache management system"""

    def __init__(self, checkpoint_cache_size: int = 500,
                 experiment_cache_size: int = 1000):
        """
        Initialize cache manager

        Args:
            checkpoint_cache_size: Size of checkpoint cache
            experiment_cache_size: Size of experiment cache
        """
        self.checkpoint_cache = CheckpointCache(checkpoint_cache_size)
        self.experiment_cache = ExperimentCache(experiment_cache_size)

    def clear_all(self) -> None:
        """Clear all caches"""
        self.checkpoint_cache.metadata_cache.clear()
        self.checkpoint_cache.data_cache.clear()
        self.checkpoint_cache.query_cache.clear()
        self.experiment_cache.cache.clear()

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        checkpoint_stats = self.checkpoint_cache.get_statistics()
        experiment_stats = self.experiment_cache.get_cache_stats()

        total_items = (checkpoint_stats['total_items'] +
                      experiment_stats['size'])

        return {
            'checkpoint_cache': checkpoint_stats,
            'experiment_cache': experiment_stats,
            'total_cached_items': total_items,
            'memory_efficiency': self._estimate_memory_efficiency()
        }

    def _estimate_memory_efficiency(self) -> Dict[str, Any]:
        """Estimate memory efficiency of caching"""
        # This is a simplified estimation
        checkpoint_stats = self.checkpoint_cache.get_statistics()
        experiment_stats = self.experiment_cache.get_cache_stats()

        total_requests = (checkpoint_stats['metadata_cache']['total_requests'] +
                         checkpoint_stats['data_cache']['total_requests'] +
                         checkpoint_stats['query_cache']['total_requests'] +
                         experiment_stats['total_requests'])

        total_hits = (checkpoint_stats['metadata_cache']['hits'] +
                     checkpoint_stats['data_cache']['hits'] +
                     checkpoint_stats['query_cache']['hits'] +
                     experiment_stats['hits'])

        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'overall_hit_rate_percent': overall_hit_rate,
            'total_requests': total_requests,
            'total_hits': total_hits,
            'estimated_db_queries_saved': total_hits
        }