#!/usr/bin/env python3
"""
动态并发控制与OOM检测模块
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class AdaptiveConcurrencyController:
    """
    自适应并发控制器
    
    功能：
    - 检测OOM事件并动态降低并发
    - 成功执行后逐步恢复并发
    - 线程安全
    """
    
    def __init__(self, max_concurrency: int = 3, min_concurrency: int = 1):
        """
        Args:
            max_concurrency: 最大并发数
            min_concurrency: 最小并发数（降级底线）
        """
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self.current_concurrency = max_concurrency
        
        self.oom_count = 0
        self.success_count = 0
        self.total_requests = 0
        
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrency)
        
        logger.info(
            f"AdaptiveConcurrencyController initialized: "
            f"max={max_concurrency}, min={min_concurrency}"
        )
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        获取并发槽位
        
        Args:
            timeout: 超时时间（秒），None=无限等待
        
        Returns:
            是否成功获取
        """
        acquired = self._semaphore.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self.total_requests += 1
        return acquired
    
    def release(self):
        """释放并发槽位"""
        self._semaphore.release()
    
    def on_oom(self):
        """OOM事件回调：降低并发"""
        with self._lock:
            self.oom_count += 1
            
            if self.current_concurrency > self.min_concurrency:
                old_concurrency = self.current_concurrency
                self.current_concurrency -= 1
                
                logger.warning(
                    f"OOM detected (total: {self.oom_count}), "
                    f"降低并发: {old_concurrency} -> {self.current_concurrency}"
                )
                
                # 更新semaphore（通过重建）
                self._rebuild_semaphore()
            else:
                logger.error(
                    f"OOM detected but already at minimum concurrency "
                    f"({self.min_concurrency})"
                )
    
    def on_success(self):
        """成功执行回调：逐步恢复并发"""
        with self._lock:
            self.success_count += 1
            
            # 策略：每10次成功恢复1个并发槽位
            if (self.success_count % 10 == 0 and 
                self.oom_count > 0 and 
                self.current_concurrency < self.max_concurrency):
                
                old_concurrency = self.current_concurrency
                self.current_concurrency = min(
                    self.current_concurrency + 1,
                    self.max_concurrency
                )
                
                logger.info(
                    f"恢复并发: {old_concurrency} -> {self.current_concurrency} "
                    f"(成功次数: {self.success_count})"
                )
                
                self._rebuild_semaphore()
    
    def _rebuild_semaphore(self):
        """重建semaphore以更新并发数"""
        # 注意：这是简化实现，生产环境可能需要更复杂的逻辑
        # 当前正在使用的semaphore不会立即生效，新请求会使用新的限制
        self._semaphore = threading.Semaphore(self.current_concurrency)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                "max_concurrency": self.max_concurrency,
                "min_concurrency": self.min_concurrency,
                "current_concurrency": self.current_concurrency,
                "total_requests": self.total_requests,
                "oom_count": self.oom_count,
                "success_count": self.success_count,
                "oom_rate": self.oom_count / max(self.total_requests, 1)
            }


class OOMDetector:
    """OOM检测器"""
    
    @staticmethod
    def is_oom_error(exception: Exception) -> bool:
        """
        判断异常是否为OOM相关
        
        Args:
            exception: 捕获的异常
        
        Returns:
            是否为OOM错误
        """
        oom_keywords = [
            'out of memory',
            'cannot allocate memory',
            'memory error',
            'allocation failed',
            'cuda out of memory',  # GPU OOM
            'std::bad_alloc'  # C++ OOM
        ]
        
        error_message = str(exception).lower()
        return any(keyword in error_message for keyword in oom_keywords)
    
    @staticmethod
    def check_available_memory() -> tuple[float, bool]:
        """
        检查可用内存
        
        Returns:
            (available_gb, is_low) 元组
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            
            # 低于2GB认为内存紧张
            is_low = available_gb < 2.0
            
            return available_gb, is_low
        except ImportError:
            # 无法检测，保守返回
            return 4.0, False


# 全局并发控制器实例
_global_controller: Optional[AdaptiveConcurrencyController] = None


def get_global_controller() -> AdaptiveConcurrencyController:
    """获取全局并发控制器（单例）"""
    global _global_controller
    if _global_controller is None:
        import os
        max_concurrency = int(os.getenv("VOXCPM_MAX_CONCURRENCY", "3"))
        _global_controller = AdaptiveConcurrencyController(max_concurrency=max_concurrency)
    return _global_controller


if __name__ == "__main__":
    """测试模式"""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    controller = AdaptiveConcurrencyController(max_concurrency=5, min_concurrency=1)
    
    print("=" * 60)
    print("动态并发控制器测试")
    print("=" * 60)
    
    # 模拟OOM事件
    for i in range(3):
        print(f"\n模拟OOM事件 #{i+1}")
        controller.on_oom()
        print(f"当前并发: {controller.current_concurrency}")
    
    # 模拟成功恢复
    print("\n模拟成功执行...")
    for i in range(25):
        controller.on_success()
        if i % 10 == 9:
            print(f"成功{i+1}次后，当前并发: {controller.current_concurrency}")
    
    # 统计
    stats = controller.get_stats()
    print("\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 内存检测
    available, is_low = OOMDetector.check_available_memory()
    print(f"\n可用内存: {available:.1f} GB")
    print(f"内存紧张: {'是' if is_low else '否'}")
    print("=" * 60)
