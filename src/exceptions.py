"""
Custom exceptions for AI Agents project
"""
from typing import Optional, Dict, Any


class AIAgentsError(Exception):
    """Base exception for AI Agents project"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        # Create a copy to avoid mutable default argument issues
        self.details = dict(details) if details is not None else {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(AIAgentsError):
    """Raised when there's a configuration error"""
    pass


class APIKeyError(AIAgentsError):
    """Raised when API key is missing or invalid"""
    pass


class ModelError(AIAgentsError):
    """Raised when there's an error with the model"""
    pass


class ToolError(AIAgentsError):
    """Raised when there's an error with a tool"""
    pass


class ResearchError(AIAgentsError):
    """Raised when there's an error during research"""
    pass


class ValidationError(AIAgentsError):
    """Raised when validation fails"""
    pass


class FileOperationError(AIAgentsError):
    """Raised when there's an error with file operations"""
    pass


class NetworkError(AIAgentsError):
    """Raised when there's a network error"""
    pass


class AgentExecutionError(AIAgentsError):
    """Raised when agent execution fails"""
    pass

