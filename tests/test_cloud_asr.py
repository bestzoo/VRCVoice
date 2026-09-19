import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.cloud_asr import CloudASR


class CloudASRTests(unittest.TestCase):
    @patch("app.cloud_asr.requests.post")
    def test_probe_posts_multipart_wav(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {"text": ""}
        post.return_value = response

        ok, _ = CloudASR.probe(
            "https://example.test/v1/audio/transcriptions", "secret", "asr-model", 9)

        self.assertTrue(ok)
        kwargs = post.call_args.kwargs
        self.assertEqual("asr-model", kwargs["data"]["model"])
        self.assertEqual("audio/wav", kwargs["files"]["file"][2])
        self.assertTrue(kwargs["files"]["file"][1].startswith(b"RIFF"))

    @patch("app.cloud_asr.requests.post")
    def test_finalize_reports_http_error_without_returning_it_as_text(self, post):
        response = Mock(status_code=400, text='{"message":"invalid"}')
        post.return_value = response
        engine = CloudASR("https://example.test", "secret", "asr-model")
        engine.accept_waveform(np.zeros(160, dtype=np.float32))

        self.assertEqual("", engine.finalize())
        self.assertIn("HTTP 400", engine.last_error)


if __name__ == "__main__":
    unittest.main()
