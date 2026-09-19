import json
import tempfile
import unittest
from pathlib import Path

from app.settings import Settings


class ModelCapabilityMigrationTests(unittest.TestCase):
    def test_old_model_entries_are_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "ai_models": {"list": [
                    {
                        "name": "ASR",
                        "endpoint": "https://example.test/v1/audio/transcriptions",
                        "model": "FunAudioLLM/SenseVoiceSmall",
                    },
                    {
                        "name": "Chat",
                        "endpoint": "https://example.test/v1/chat/completions",
                        "model": "chat-model",
                    },
                ]}
            }), encoding="utf-8")

            settings = Settings(str(path)).load()

            self.assertEqual(["ASR"], settings.ai_model_names("speech"))
            self.assertEqual(["Chat"], settings.ai_model_names("text"))


if __name__ == "__main__":
    unittest.main()
