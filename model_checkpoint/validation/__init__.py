"""
PE-VIT Ecosystem Validation Module

Provides validation functionality for the entire PE-VIT ecosystem including:
- System structure validation
- Database integrity checks
- Experiment lifecycle testing
- Data directory validation
- Symlink verification
"""

from .system_validator import SystemValidator, ValidationResult

__all__ = ["SystemValidator", "ValidationResult"]