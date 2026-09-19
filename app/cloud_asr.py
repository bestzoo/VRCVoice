"""云端 ASR 后端: 调用 OpenAI 兼容的 /v1/audio/transcriptions 接口。
适用于空间紧张/低配机器(本地模型约 500MB, 云端不占空间)。
接口兼容: 硅基流动 SiliconFlow / OpenAI / 各类中转站。
"""
import io
import wave
import numpy as np
import requests


class CloudASR:
    def __init__(self, endpoint: str, api_key: str, model: str,
                 language: str = "", timeout_sec: int = 30):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.language = language
        self.timeout = timeout_sec
        self._chunks = []
        self.last_error = ""

    def reset(self):
        self._chunks = []
        self.last_error = ""

    def accept_waveform(self, samples: np.ndarray) -> str:
        """云端不流式, 只攒音频, 返回空串。"""
        if samples is not None and samples.size > 0:
            self._chunks.append(samples)
        return ""

    def finalize(self) -> str:
        if not self._chunks:
            return ""
        if not self.api_key:
            self.last_error = "未配置云端 API Key"
            return ""
        audio = np.concatenate(self._chunks)
        wav_bytes = self._to_wav(audio)
        try:
            return self.transcribe_wav(wav_bytes)
        except Exception as e:
            self.last_error = f"云端请求失败: {e}"
            return ""

    def transcribe_wav(self, wav_bytes: bytes) -> str:
        """发送一个 WAV 到真实转写接口；失败时抛异常，供识别和连通测试复用。"""
        if not self.endpoint:
            raise ValueError("未配置语音识别 API 地址")
        if not self.api_key:
            raise ValueError("未配置云端 API Key")
        if not self.model:
            raise ValueError("未配置语音识别模型名")
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model}
        if self.language:
            data["language"] = self.language
        resp = requests.post(
            self.endpoint, headers={"Authorization": f"Bearer {self.api_key}"},
            files=files, data=data, timeout=self.timeout,
        )
        if not 200 <= resp.status_code < 300:
            body = (resp.text or "").replace("\r", " ").replace("\n", " ")[:200]
            raise RuntimeError(f"云端返回 HTTP {resp.status_code}: {body}")
        try:
            out = resp.json()
        except ValueError as e:
            raise RuntimeError("语音接口返回的不是 JSON") from e
        if not isinstance(out, dict):
            raise RuntimeError("语音接口返回格式无效")
        return str(out.get("text", "") or "").strip()

    @classmethod
    def probe(cls, endpoint: str, api_key: str, model: str,
              timeout_sec: int = 15) -> tuple:
        """用短静音 WAV 实测语音转写接口，而不是错误地发送聊天 JSON。"""
        engine = cls(endpoint, api_key, model, timeout_sec=timeout_sec)
        samples = np.zeros(8000, dtype=np.float32)  # 0.5 秒、16 kHz 合法 WAV
        engine.transcribe_wav(cls._to_wav(samples))
        return True, "语音识别接口请求成功"

    @staticmethod
    def _to_wav(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            w.writeframes(pcm.tobytes())
        return buf.getvalue()
