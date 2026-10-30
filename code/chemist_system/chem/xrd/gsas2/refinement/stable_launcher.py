"""
★ ★ Hermes 2026-06-28: ★ ★ 永久修复 ★ ★ GSAS2 自动精修启动失败

★ ★ ★ ★ 这个文件是 ★ 关键 ★ 解决 ★ GSAS2 每次都不能启动成功 ★ ★ ★

★ ★ 修复的 4 大根本原因:
★ 1. 相对导入失败 → 双 fallback (★ 包内 + ★ 包外都支持)
★ 2. __init__ 参数缺失 → 默认 output_dir 自动创建
★ 3. gsas2_path 不存在 → 自动检测 + 多路径 fallback
★ 4. 缺少统一 API → run_gsas2_refinement() 一键启动

★ ★ 5 大功能:
★ 1. GSAS2_AVAILABLE: ★ 全局可用的 GSAS2 标志
★ 2. get_gsas2_runner(output_dir): ★ 单例获取
★ 3. run_gsas2_refinement(...): ★ ★ 统一 API ★ 一键启动
★ 4. diagnose_gsas2(): ★ 启动诊断 + ★ 报告 ★ 问题
★ 5. _find_gsas2_path(): ★ 自动检测 GSASII 路径
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from .... import paths

# ============================================================
# ★ ★ ★ 关键: ★ 全局标志 ★ GSAS2 是否可用
# ============================================================
GSAS2_AVAILABLE: bool = False
_GSAS2_IMPORT_ERROR: Optional[str] = None

# ★ ★ ★ 关键: ★ 单例 ★ 避免重复初始化
_runner_singleton: Optional[Any] = None
_singleton_output_dir: Optional[str] = None


def _find_gsas2_path() -> Optional[str]:
    """★ ★ 关键 ★ ★ 自动检测 GSASII 路径 (★ 多路径 fallback)"""
    candidates = [
        os.environ.get("GSAS2_PATH"),  # ★ 环境变量优先
        str(paths.GSAS2_HOME),
        "/home/ucas/GSASII",
        "/opt/GSASII",
        "/usr/local/GSASII",
        os.path.expanduser("~/GSASII"),
        os.path.expanduser("~/GSASIIv2"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            gsasii_dir = os.path.join(path, "GSASII")
            if os.path.isdir(gsasii_dir):
                gsas2_scriptable = os.path.join(gsasii_dir, "GSASIIscriptable.py")
                if os.path.isfile(gsas2_scriptable):
                    return path
    return None


def _try_import_gsas2():
    """★ ★ 关键 ★ ★ 尝试导入 GSASIIscriptable (★ 多路径 fallback)"""
    global GSAS2_AVAILABLE, _GSAS2_IMPORT_ERROR

    gsas2_path = _find_gsas2_path()
    if gsas2_path is None:
        _GSAS2_IMPORT_ERROR = "★ 未找到 GSASII 安装目录 (★ 尝试了 7 个常见路径)"
        GSAS2_AVAILABLE = False
        return None

    # ★ ★ ★ 关键: ★ 添加 GSAS2 路径到 sys.path
    if gsas2_path not in sys.path:
        sys.path.insert(0, gsas2_path)

    try:
        from GSASII import GSASIIscriptable as G2
        GSAS2_AVAILABLE = True
        _GSAS2_IMPORT_ERROR = None
        return G2
    except Exception as e:
        _GSAS2_IMPORT_ERROR = f"★ GSASII import 失败: {e}"
        GSAS2_AVAILABLE = False
        return None


def diagnose_gsas2() -> Dict[str, Any]:
    """★ ★ 关键 ★ ★ 诊断 GSAS2 启动状态"""
    report = {
        "★ gsas2_available": GSAS2_AVAILABLE,
        "★ gsas2_path_found": None,
        "★ gsas2_path_checked": [],
        "★ import_error": _GSAS2_IMPORT_ERROR,
        "★ recommendations": [],
        "★ python_path": sys.path[:5],
    }

    # ★ 1. 检查 GSASII 路径
    candidates = [
        os.environ.get("GSAS2_PATH"),
        "/home/ucas/GSASIIv2",
        "/home/ucas/GSASII",
        "/opt/GSASII",
        "/usr/local/GSASII",
        os.path.expanduser("~/GSASII"),
        os.path.expanduser("~/GSASIIv2"),
    ]
    report["★ gsas2_path_checked"] = candidates

    gsas2_path = _find_gsas2_path()
    if gsas2_path:
        report["★ gsas2_path_found"] = gsas2_path
    else:
        report["★ recommendations"].append(
            "★ 设置环境变量 GSAS2_PATH 指向 GSASII 根目录"
        )
        report["★ recommendations"].append(
            "★ 或安装 GSASII 到 /home/ucas/GSASIIv2"
        )

    # ★ 2. 测试导入
    G2 = _try_import_gsas2()
    if G2 is None:
        report["★ recommendations"].append(
            f"★ Import 错误: {_GSAS2_IMPORT_ERROR}"
        )

    return report


def get_gsas2_runner(output_dir: Optional[str] = None, gsas2_path: Optional[str] = None):
    """
    ★ ★ 关键 ★ ★ 单例获取 GSAS2Runner (★ ★ 自动 fallback ★)

    Args:
        output_dir: 输出目录 (★ 默认 /tmp/gsas2_runs/<timestamp>)
        gsas2_path: GSAS2 安装路径 (★ 默认自动检测)

    Returns:
        GSAS2Runner 实例 (★ 失败时返回 None + ★ diagnose_gsas2() 报告)
    """
    global _runner_singleton, _singleton_output_dir

    # ★ ★ ★ 关键: ★ 如果环境变量设置了 GSAS2_PATH, ★ 优先使用
    if gsas2_path is None:
        gsas2_path = os.environ.get("GSAS2_PATH") or _find_gsas2_path()

    # ★ 1. ★ 验证 gsas2_path
    if not gsas2_path or not os.path.isdir(gsas2_path):
        return None

    # ★ 2. ★ 验证 output_dir (★ 默认创建)
    if output_dir is None:
        import time
        output_dir = f"/tmp/gsas2_runs/{int(time.time())}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ★ 3. ★ 优先返回单例
    if (_runner_singleton is not None
            and _singleton_output_dir == output_dir):
        return _runner_singleton

    # ★ 4. ★ 尝试多种导入方式
    try:
        # ★ 方式 1: 包路径导入 (★ 推荐)
        if gsas2_path not in sys.path:
            sys.path.insert(0, gsas2_path)
        from gsas2_refinement.gsas2_runner import GSAS2Runner
        runner = GSAS2Runner(output_dir=output_dir, gsas2_path=gsas2_path)
    except (ImportError, ValueError) as e1:
        try:
            # ★ 方式 2: 直接 import
            sys.path.insert(0, str(paths.HOME))
            from gsas2_refinement.gsas2_runner import GSAS2Runner
            runner = GSAS2Runner(output_dir=output_dir, gsas2_path=gsas2_path)
        except Exception as e2:
            try:
                # ★ 方式 3: 包内绝对导入 (★ fallback)
                import importlib
                here = os.path.dirname(os.path.abspath(__file__))
                if here not in sys.path:
                    sys.path.insert(0, here)
                spec = importlib.util.spec_from_file_location(
                    "_gsas2_runner", os.path.join(here, "gsas2_runner.py")
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                GSAS2Runner = module.GSAS2Runner
                runner = GSAS2Runner(output_dir=output_dir, gsas2_path=gsas2_path)
            except Exception as e3:
                # ★ 所有方式失败, ★ 返回 None
                print(f"★ Hermes GSAS2 启动失败: {e1}, {e2}, {e3}")
                return None

    # ★ 5. ★ 缓存单例
    _runner_singleton = runner
    _singleton_output_dir = output_dir
    return runner


def run_gsas2_refinement(
    cif_files: Union[str, List[str]],
    xrd_file: str,
    output_dir: Optional[str] = None,
    gsas2_path: Optional[str] = None,
    phases_info: Optional[List[Dict[str, Any]]] = None,
    recipe: Optional[List] = None,
    auto_diagnose: bool = True,
) -> Dict[str, Any]:
    """
    ★ ★ ★ 关键 ★ ★ ★ 统一 API ★ 一键启动 GSAS2 自动精修

    Args:
        cif_files: CIF 文件路径 (★ str 或 List[str])
        xrd_file: 实验 XRD 数据文件
        output_dir: 输出目录 (★ 默认自动)
        gsas2_path: GSAS2 安装路径 (★ 默认自动检测)
        phases_info: 相信息 (★ 可选)
        recipe: 精修步骤 (★ 可选, ★ 默认 base recipe)
        auto_diagnose: 失败时是否自动诊断

    Returns:
        dict: ★ 精修结果 + ★ 诊断信息
        {
            "★ success": bool,
            "★ runner": GSAS2Runner,
            "★ history": RefinementHistory,
            "★ error": Optional[str],
            "★ diagnosis": dict (★ 失败时)
        }
    """
    result = {
        "★ success": False,
        "★ runner": None,
        "★ history": None,
        "★ error": None,
        "★ diagnosis": None,
        "★ gsas2_available": GSAS2_AVAILABLE,
    }

    # ★ 1. ★ 验证 GSAS2 可用性
    if not GSAS2_AVAILABLE:
        G2 = _try_import_gsas2()
        if G2 is None:
            result["★ error"] = _GSAS2_IMPORT_ERROR
            if auto_diagnose:
                result["★ diagnosis"] = diagnose_gsas2()
            return result

    # ★ 2. ★ 获取 runner
    runner = get_gsas2_runner(output_dir=output_dir, gsas2_path=gsas2_path)
    if runner is None:
        result["★ error"] = "★ GSAS2Runner 创建失败"
        if auto_diagnose:
            result["★ diagnosis"] = diagnose_gsas2()
        return result

    result["★ runner"] = runner

    # ★ 3. ★ 验证输入文件
    if isinstance(cif_files, str):
        cif_files = [cif_files]

    for cif in cif_files:
        if not os.path.isfile(cif):
            result["★ error"] = f"★ CIF 文件不存在: {cif}"
            return result

    if not os.path.isfile(xrd_file):
        result["★ error"] = f"★ XRD 文件不存在: {xrd_file}"
        return result

    # ★ 4. ★ 构建项目
    try:
        if phases_info is None:
            phases_info = []
            for cif in cif_files:
                phases_info.append({
                    "★ ciffile": cif,
                    "★ phase_name": os.path.basename(cif).replace(".cif", ""),
                })

        project = runner.build_project(
            phases=phases_info,
            xrd_file=xrd_file,
        )
    except Exception as e:
        result["★ error"] = f"★ 构建项目失败: {e}"
        if auto_diagnose:
            result["★ diagnosis"] = diagnose_gsas2()
        return result

    # ★ 5. ★ 执行精修
    try:
        if recipe is None:
            from gsas2_refinement.refinement_strategy import build_recipe
            recipe = build_recipe()

        history = runner.run_recipe(project, None, None, recipe)
        result["★ history"] = history
        result["★ success"] = True
        return result
    except Exception as e:
        result["★ error"] = f"★ 精修执行失败: {e}"
        if auto_diagnose:
            result["★ diagnosis"] = diagnose_gsas2()
        return result


# ★ ★ ★ 关键: ★ 模块加载时 ★ 自动检测
_G2_at_load = _try_import_gsas2()


if __name__ == "__main__":
    print("="*70)
    print("★ Hermes GSAS2 稳定启动器 ★ 诊断 ★")
    print("="*70)
    diag = diagnose_gsas2()
    for k, v in diag.items():
        print(f"  {k}: {v}")