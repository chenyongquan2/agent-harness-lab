"""离线演示脚本不能悄悄坏掉：README 里把它当作不需要密钥的入口。"""

import asyncio
import importlib.util
from pathlib import Path


def test_offline_demo_runs_and_shows_the_three_cases(capsys):
    path = Path(__file__).resolve().parent.parent / "scripts" / "demo_offline.py"
    spec = importlib.util.spec_from_file_location("demo_offline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    asyncio.run(module.main())

    out = capsys.readouterr().out
    assert "已预约" in out or "预约号 A-001" in out  # 正常完成
    assert "工具执行失败" in out  # 工具出错被回灌，循环没有崩
    assert "spin_detected" in out  # 原地重复被护栏停止
