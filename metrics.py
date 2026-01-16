#!/usr/bin/env python3
"""
性能监控与指标追踪模块
"""
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    request_id: str
    text_length: int
    audio_duration: float  # 秒
    total_time: float  # 秒
    rtf: float = 0.0  # Real-Time Factor，默认0（未计算）
    
    # 模块耗时（秒）
    text_embed_time: float = 0.0
    vae_encode_time: float = 0.0
    feat_encode_time: float = 0.0
    decode_loop_time: float = 0.0
    vae_decode_time: float = 0.0
    
    # 其他信息
    model_version: str = "unknown"
    voice: str = "default"
    fixed_timesteps: int = 10
    num_decode_steps: int = 0
    
    # 时间戳
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def finalize(self):
        """完成指标收集"""
        self.end_time = time.time()
        if self.total_time == 0:
            self.total_time = self.end_time - self.start_time
        if self.audio_duration > 0:
            self.rtf = self.total_time / self.audio_duration
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "text_length": self.text_length,
            "audio_duration": self.audio_duration,
            "total_time": round(self.total_time, 3),
            "rtf": round(self.rtf, 3),
            "timeline": {
                "text_embed": round(self.text_embed_time, 3),
                "vae_encode": round(self.vae_encode_time, 3),
                "feat_encode": round(self.feat_encode_time, 3),
                "decode_loop": round(self.decode_loop_time, 3),
                "vae_decode": round(self.vae_decode_time, 3),
            },
            "model": {
                "version": self.model_version,
                "voice": self.voice,
                "fixed_timesteps": self.fixed_timesteps,
                "num_decode_steps": self.num_decode_steps
            },
            "timestamp": {
                "start": self.start_time,
                "end": self.end_time or time.time()
            }
        }
    
    def log_structured(self):
        """输出结构化日志"""
        logger.info(
            "tts_metrics",
            extra={"metrics": self.to_dict()}
        )


class MetricsCollector:
    """指标收集器（全局单例）"""
    
    def __init__(self):
        self.requests: list[PerformanceMetrics] = []
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        # 聚合统计
        self._rtf_values: list[float] = []
        self._latency_values: list[float] = []
    
    def record(self, metrics: PerformanceMetrics, success: bool = True):
        """记录一次请求的指标"""
        self.requests.append(metrics)
        self.request_count += 1
        
        if success:
            self.success_count += 1
            self._rtf_values.append(metrics.rtf)
            self._latency_values.append(metrics.total_time)
        else:
            self.failure_count += 1
        
        # 限制内存占用（只保留最近1000条）
        if len(self.requests) > 1000:
            self.requests = self.requests[-1000:]
        if len(self._rtf_values) > 1000:
            self._rtf_values = self._rtf_values[-1000:]
            self._latency_values = self._latency_values[-1000:]
    
    def get_summary(self) -> dict:
        """获取统计摘要"""
        if not self._rtf_values:
            return {
                "total_requests": self.request_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": 0.0
            }
        
        rtf_sorted = sorted(self._rtf_values)
        latency_sorted = sorted(self._latency_values)
        
        def percentile(values: list[float], p: float) -> float:
            idx = int(len(values) * p)
            return values[min(idx, len(values) - 1)]
        
        return {
            "total_requests": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / max(self.request_count, 1),
            "rtf": {
                "mean": sum(self._rtf_values) / len(self._rtf_values),
                "p50": percentile(rtf_sorted, 0.5),
                "p95": percentile(rtf_sorted, 0.95),
                "p99": percentile(rtf_sorted, 0.99),
                "min": min(self._rtf_values),
                "max": max(self._rtf_values)
            },
            "latency": {
                "mean": sum(self._latency_values) / len(self._latency_values),
                "p50": percentile(latency_sorted, 0.5),
                "p95": percentile(latency_sorted, 0.95),
                "p99": percentile(latency_sorted, 0.99),
                "min": min(self._latency_values),
                "max": max(self._latency_values)
            }
        }
    
    def reset(self):
        """重置统计"""
        self.requests.clear()
        self._rtf_values.clear()
        self._latency_values.clear()
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0


class TimedSection:
    """
    计时上下文管理器
    
    Example:
        with TimedSection("my_operation") as timer:
            do_something()
        print(f"Took {timer.elapsed:.3f}s")
    """
    
    def __init__(self, name: str, log_on_exit: bool = False):
        self.name = name
        self.log_on_exit = log_on_exit
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        if self.log_on_exit:
            logger.debug(f"{self.name} took {self.elapsed:.3f}s")


# 全局指标收集器
_global_collector: Optional[MetricsCollector] = None


def get_global_collector() -> MetricsCollector:
    """获取全局指标收集器（单例）"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


if __name__ == "__main__":
    """测试模式"""
    import uuid
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    print("=" * 60)
    print("性能监控测试")
    print("=" * 60)
    
    collector = MetricsCollector()
    
    # 模拟10次请求
    for i in range(10):
        metrics = PerformanceMetrics(
            request_id=str(uuid.uuid4()),
            text_length=50 + i * 10,
            audio_duration=3.0,
            total_time=0.0
        )
        
        # 模拟各阶段耗时
        with TimedSection("text_embed") as timer:
            time.sleep(0.01)
        metrics.text_embed_time = timer.elapsed
        
        with TimedSection("vae_encode") as timer:
            time.sleep(0.02)
        metrics.vae_encode_time = timer.elapsed
        
        with TimedSection("decode_loop") as timer:
            time.sleep(0.5 + i * 0.05)  # 递增耗时
        metrics.decode_loop_time = timer.elapsed
        
        metrics.finalize()
        collector.record(metrics, success=True)
    
    # 统计摘要
    summary = collector.get_summary()
    print("\n摘要:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # 最后一次请求详情
    last_metrics = collector.requests[-1]
    print("\n最后一次请求详情:")
    print(json.dumps(last_metrics.to_dict(), indent=2, ensure_ascii=False))
    
    print("=" * 60)
