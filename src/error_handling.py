"""
Error handling utilities and decorators
"""
import logging
import functools
import traceback
import time
from typing import Callable, Any, Optional, Type, Union
from .exceptions import AIAgentsError


logger = logging.getLogger(__name__)


def handle_errors(
    default_return: Any = None,
    raise_exception: bool = False,
    log_traceback: bool = True,
    exception_type: Optional[Type[Exception]] = None
) -> Callable[[Callable], Callable]:
    """
    Decorator for error handling
    
    Args:
        default_return: Value to return on error (if not raising)
        raise_exception: Whether to raise exception or return default
        log_traceback: Whether to log full traceback
        exception_type: Custom exception type to raise
    
    Returns:
        Decorator function
    
    Example:
        @handle_errors(default_return=None, raise_exception=False)
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AIAgentsError as e:
                func_name = getattr(func, '__name__', str(func))
                logger.error(f"Error in {func_name}: {e.message}")
                if e.details:
                    logger.debug(f"Error details: {e.details}")
                if log_traceback:
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                if raise_exception:
                    raise
                return default_return
            except Exception as e:
                func_name = getattr(func, '__name__', str(func))
                error_msg = f"Unexpected error in {func_name}: {str(e)}"
                logger.error(error_msg)
                if log_traceback:
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if exception_type:
                    raise exception_type(error_msg) from e
                elif raise_exception:
                    raise
                return default_return
        
        return wrapper
    return decorator


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    log_retries: bool = True
) -> Callable[[Callable], Callable]:
    """
    Decorator to retry function on error
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        log_retries: Whether to log retry attempts
    
    Returns:
        Decorator function
    
    Example:
        @retry_on_error(max_retries=3, delay=1.0)
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            func_name = getattr(func, '__name__', str(func))
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if log_retries:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries + 1} failed for {func_name}: {e}. "
                                f"Retrying in {current_delay}s..."
                            )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        if log_retries:
                            logger.error(f"All {max_retries + 1} attempts failed for {func_name}")
                        raise
            
            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    log_error: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default_return: Value to return on error
        log_error: Whether to log errors
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            func_name = getattr(func, '__name__', str(func))
            logger.error(f"Error executing {func_name}: {e}")
        return default_return


def validate_not_none(value: Any, name: str, error_type: Type[Exception] = ValueError):
    """
    Validate that a value is not None
    
    Args:
        value: Value to validate
        name: Name of the value (for error message)
        error_type: Exception type to raise
    
    Raises:
        error_type: If value is None
    """
    if value is None:
        raise error_type(f"{name} cannot be None")


def validate_not_empty(value: Any, name: str, error_type: Type[Exception] = ValueError):
    """
    Validate that a value is not empty
    
    Args:
        value: Value to validate
        name: Name of the value (for error message)
        error_type: Exception type to raise
    
    Raises:
        error_type: If value is empty
    """
    if not value:
        raise error_type(f"{name} cannot be empty")


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log execution time of a function
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = getattr(func, '__name__', str(func))
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func_name} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func_name} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper

