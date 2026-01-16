#!/usr/bin/env python3
"""
超时与重试机制模块
"""
import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TimeoutError(Exception):
    """超时异常"""
    pass


class RetryableError(Exception):
    """可重试的异常"""
    pass


def with_timeout(timeout_seconds: float):
    """
    超时装饰器（同步函数）
    
    Args:
        timeout_seconds: 超时时间（秒）
    
    Example:
        @with_timeout(30.0)
        def slow_function():
            time.sleep(100)  # 会在30秒后抛出TimeoutError
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")
            
            # 设置信号处理器
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                # 恢复原处理器
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        return wrapper
    return decorator


def with_async_timeout(timeout_seconds: float):
    """
    异步超时装饰器
    
    Args:
        timeout_seconds: 超时时间（秒）
    
    Example:
        @with_async_timeout(30.0)
        async def slow_async_function():
            await asyncio.sleep(100)  # 会在30秒后抛出TimeoutError
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Async function {func.__name__} timed out after {timeout_seconds}s"
                )
        
        return wrapper
    return decorator


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    retry_on: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    重试装饰器（指数退避）
    
    Args:
        max_attempts: 最大尝试次数
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数基数（2 = 1s, 2s, 4s, 8s...）
        retry_on: 需要重试的异常类型
        on_retry: 重试回调函数(exception, attempt)
    
    Example:
        @with_retry(max_attempts=3, retry_on=(ConnectionError,))
        def unstable_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    # 计算延迟（指数退避）
                    delay = min(
                        initial_delay * (exponential_base ** (attempt - 1)),
                        max_delay
                    )
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(delay)
            
            raise last_exception  # 理论上不会到达这里
        
        return wrapper
    return decorator


def with_async_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    retry_on: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """异步重试装饰器"""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    delay = min(
                        initial_delay * (exponential_base ** (attempt - 1)),
                        max_delay
                    )
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    断路器模式（可选，高级功能）
    
    当连续失败次数超过阈值时，暂时停止请求，避免雪崩
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Args:
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时（秒）
            expected_exception: 计数的异常类型
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """通过断路器调用函数"""
        if self.state == "open":
            # 检查是否可以恢复
            if self.last_failure_time and \
               (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN, request rejected")
        
        try:
            result = func(*args, **kwargs)
            
            # 成功，重置
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed (recovered)")
            
            return result
        
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    f"Circuit breaker OPEN after {self.failure_count} failures"
                )
            
            raise


if __name__ == "__main__":
    """测试模式"""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("超时与重试测试")
    print("=" * 60)
    
    # 测试重试
    @with_retry(max_attempts=3, initial_delay=0.5, retry_on=(ValueError,))
    def flaky_function(should_succeed: bool = False):
        if not should_succeed:
            raise ValueError("Intentional failure")
        return "Success!"
    
    print("\n测试1: 最终成功的重试")
    counter = [0]
    
    @with_retry(max_attempts=3, initial_delay=0.1)
    def eventually_succeeds():
        counter[0] += 1
        if counter[0] < 3:
            raise ValueError(f"Attempt {counter[0]} failed")
        return "Success!"
    
    result = eventually_succeeds()
    print(f"结果: {result}")
    print(f"尝试次数: {counter[0]}")
    
    # 测试断路器
    print("\n测试2: 断路器")
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)
    
    def failing_func():
        raise ValueError("Always fails")
    
    for i in range(5):
        try:
            breaker.call(failing_func)
        except Exception as e:
            print(f"尝试{i+1}: {e}")
    
    print(f"断路器状态: {breaker.state}")
    print("=" * 60)
