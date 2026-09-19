import threading
import sys
import types
import unittest
from unittest.mock import Mock

import numpy as np

# 这个单元测试只覆盖 stop() 的错误分流，不需要加载声卡、Sherpa 或 OSC 运行库。
for module_name, class_name in (("app.recorder", "Recorder"),
                                ("app.asr_engine", "ASREngine"),
                                ("app.output", "OutputManager")):
    module = types.ModuleType(module_name)
    setattr(module, class_name, type(class_name, (), {}))
    sys.modules.setdefault(module_name, module)
vrc_status = types.ModuleType("app.vrc_status")
vrc_status.vrc_ok = lambda: True
sys.modules.setdefault("app.vrc_status", vrc_status)

from app.controller import RecognitionController


class ControllerErrorOutputTests(unittest.TestCase):
    def test_asr_error_is_not_sent_to_vrchat(self):
        controller = RecognitionController.__new__(RecognitionController)
        controller._lock = threading.Lock()
        controller._active = True
        controller._partial_text = ""
        controller._last_text = ""
        controller.recorder = Mock()
        controller.recorder.stop.return_value = np.zeros(160, dtype=np.float32)
        controller.asr = Mock(last_error="云端返回 HTTP 400")
        controller.asr.finalize.return_value = ""
        controller.output = Mock()
        controller.settings = Mock()
        controller.on_state_changed = None
        controller.on_finished = Mock()
        controller.on_message_logged = None

        controller.stop()

        controller.output.send.assert_not_called()
        controller.on_finished.assert_called_once_with("[错误] 云端返回 HTTP 400")


if __name__ == "__main__":
    unittest.main()
