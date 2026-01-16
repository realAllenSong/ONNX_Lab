#!/usr/bin/env python3
"""
自动模型选择模块
根据系统环境（平台、CPU架构、内存）自动选择最优的VoxCPM模型版本
"""
import platform
import sys
from typing import Literal, Optional

try:
    import psutil
except ImportError:
    psutil = None

ModelVersion = Literal["0.5B", "1.5B"]
ModelPrecision = Literal["fp32", "int8"]


class ModelSelector:
    """自动模型版本选择器"""
    
    def __init__(self):
        self.system = platform.system()  # 'Darwin', 'Linux', 'Windows'
        self.machine = platform.machine()  # 'arm64', 'x86_64', 'AMD64'
        self.memory_gb = self._get_memory_gb()
    
    def _get_memory_gb(self) -> float:
        """获取系统总内存（GB）"""
        if psutil is not None:
            return psutil.virtual_memory().total / (1024 ** 3)
        
        # fallback: 保守估计
        if self.system == 'Darwin':
            return 8.0  # Mac最少8GB
        return 4.0  # 其他平台保守估计
    
    def select_version(
        self,
        prefer_quality: bool = True,
        force_version: Optional[ModelVersion] = None,
        force_precision: Optional[ModelPrecision] = None
    ) -> tuple[ModelVersion, ModelPrecision]:
        """
        自动选择模型版本和精度
        
        Args:
            prefer_quality: True=优先质量（1.5B），False=优先速度（0.5B）
            force_version: 强制指定版本（覆盖自动选择）
            force_precision: 强制指定精度（覆盖自动选择）
        
        Returns:
            (version, precision) 元组
        """
        # 环境变量覆盖
        import os
        env_version = os.getenv("VOXCPM_MODEL_VERSION")
        env_precision = os.getenv("VOXCPM_MODEL_PRECISION")
        
        if env_version in ("0.5B", "1.5B"):
            force_version = env_version  # type: ignore
        if env_precision in ("fp32", "int8"):
            force_precision = env_precision  # type: ignore
        
        # 如果完全手动指定，直接返回
        if force_version and force_precision:
            return force_version, force_precision
        
        # 自动选择
        version = force_version or self._select_version(prefer_quality)
        precision = force_precision or self._select_precision(version)
        
        return version, precision
    
    def _select_version(self, prefer_quality: bool) -> ModelVersion:
        """根据内存选择版本"""
        if self.memory_gb >= 12 and prefer_quality:
            return "1.5B"
        elif self.memory_gb >= 6:
            return "0.5B" if not prefer_quality else "1.5B"
        else:
            # 低内存设备强制0.5B
            return "0.5B"
    
    def _select_precision(self, version: ModelVersion) -> ModelPrecision:
        """
        根据平台和版本选择精度
        
        策略：
        - Mac M系列（ARM64）→ 量化（int8）优先
        - Linux/Windows → 全精度（fp32）优先，避免量化失真
        - 低内存设备 → 必须量化
        """
        # 低内存强制量化
        if self.memory_gb < 6:
            return "int8"
        
        # Mac M系列优先量化
        if self.system == 'Darwin' and self.machine == 'arm64':
            return "int8"
        
        # 其他平台优先全精度（避免Linux量化失真问题）
        return "fp32"
    
    def get_model_dir_name(self, version: ModelVersion, precision: ModelPrecision) -> str:
        """生成模型目录名"""
        if precision == "int8":
            return f"onnx_models_quantized_{version.lower()}"
        else:
            return f"onnx_models_{version.lower()}"
    
    def get_summary(self) -> dict:
        """获取检测摘要"""
        version, precision = self.select_version()
        return {
            "system": self.system,
            "machine": self.machine,
            "memory_gb": round(self.memory_gb, 1),
            "selected_version": version,
            "selected_precision": precision,
            "model_dir": self.get_model_dir_name(version, precision),
            "estimated_memory_usage_gb": self._estimate_memory_usage(version, precision)
        }
    
    def _estimate_memory_usage(self, version: ModelVersion, precision: ModelPrecision) -> float:
        """估算模型内存占用（GB）"""
        base_usage = {
            ("0.5B", "fp32"): 2.0,
            ("0.5B", "int8"): 0.5,
            ("1.5B", "fp32"): 6.0,
            ("1.5B", "int8"): 1.5
        }
        return base_usage.get((version, precision), 2.0)
    
    def check_compatibility(self) -> tuple[bool, list[str]]:
        """
        检查系统兼容性
        
        Returns:
            (is_compatible, warnings) 元组
        """
        warnings = []
        
        # 检查内存
        if self.memory_gb < 4:
            warnings.append(f"内存不足（{self.memory_gb:.1f}GB < 4GB），可能无法运行")
        
        # 检查CPU指令集（需要AVX2）
        if self.system != 'Darwin':  # Mac一般都支持
            cpu_features = self._check_cpu_features()
            if 'avx2' not in cpu_features:
                warnings.append("CPU可能不支持AVX2指令集，ONNX推理可能失败")
        
        # Windows未测试警告
        if self.system == 'Windows':
            warnings.append("Windows平台未充分测试，可能遇到兼容性问题")
        
        is_compatible = len(warnings) == 0 or all('可能' in w for w in warnings)
        return is_compatible, warnings
    
    def _check_cpu_features(self) -> set[str]:
        """检测CPU特性（简化版）"""
        features = set()
        
        try:
            if self.system == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('flags'):
                            features.update(line.split(':')[1].strip().split())
                            break
            # TODO: Windows/Mac的CPU特性检测
        except Exception:
            pass
        
        return features


def auto_select_model(prefer_quality: bool = True) -> tuple[ModelVersion, ModelPrecision]:
    """
    便捷函数：自动选择模型
    
    Args:
        prefer_quality: True=优先质量（1.5B），False=优先速度（0.5B）
    
    Returns:
        (version, precision) 元组
    """
    selector = ModelSelector()
    return selector.select_version(prefer_quality=prefer_quality)


if __name__ == "__main__":
    """诊断模式"""
    selector = ModelSelector()
    summary = selector.get_summary()
    compatible, warnings = selector.check_compatibility()
    
    print("=" * 60)
    print("VoxCPM ONNX 模型自动选择诊断")
    print("=" * 60)
    print(f"系统平台: {summary['system']}")
    print(f"CPU架构: {summary['machine']}")
    print(f"系统内存: {summary['memory_gb']} GB")
    print()
    print(f"推荐模型: VoxCPM {summary['selected_version']} ({summary['selected_precision'].upper()})")
    print(f"模型目录: {summary['model_dir']}")
    print(f"预估内存: {summary['estimated_memory_usage_gb']} GB")
    print()
    
    if compatible:
        print("✅ 系统兼容性检查通过")
    else:
        print("⚠️  系统兼容性警告:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print()
    print("环境变量覆盖:")
    print("  export VOXCPM_MODEL_VERSION=0.5B|1.5B")
    print("  export VOXCPM_MODEL_PRECISION=fp32|int8")
    print("=" * 60)
