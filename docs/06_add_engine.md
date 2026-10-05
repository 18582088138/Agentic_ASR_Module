# 06 · 怎么接一个新 ASR 引擎 / Adding an engine

> **什么时候读**：要给本模块加一个新引擎（另一个本地模型、另一家云 API、内部自研）时。

---

## 1 · 抽象面有多窄

一个引擎只需要会做**两件事**：

| 必实现 | 作用 |
|---|---|
| `transcribe(chunk, *, language, word_timestamps)` | 一段音频 → 一份 `Transcript` |
| `declared_capabilities()`（classmethod） | 静态声明自己会什么 |

**为什么这么窄**：切片与时间轴平移、字幕渲染、幻觉闸门、情绪补齐、轨道合并这五件事
**与引擎无关**。一旦下沉到各引擎，就会出现 N 份互相偷偷不一致的实现 ——
同一段音频换个引擎得到不同切法，调试时看不出是哪一份在起作用。
所以它们全部留在门面 `ASRModule`（`asr.py`）里，对所有引擎共享。

于是新增引擎的边际成本就是那两件事。可选实现：`detect_emotion()`、`languages()`、
`release()`、`info()` —— 都有默认实现。

---

## 2 · 四步

### ① 建文件 `agentic_asr/engines/myengine.py`

```python
"""我的引擎 / My engine.

一句话说清它是什么、在哪跑、有什么特别的取舍。
"""

from __future__ import annotations

from typing import Any

from agentic_asr.core.errors import EngineError
from agentic_asr.core.types import AudioChunk, Capability, Segment, Transcript
from agentic_asr.engines.base import ASREngine


class MyEngine(ASREngine):
    name = "myengine"
    implemented = True
    # 单次调用能吃的最长音频（秒）。None = 本地不限，门面就不切了。
    max_audio_seconds: float | None = 120.0

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._model = None                      # 懒加载：构造时必须瞬时

    # ── 能力：必须不加载权重就能回答 ──────────────────────────────────────
    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        return {Capability.TIMESTAMPS, Capability.LANGUAGE_ID}

    def languages(self) -> list[str]:
        return ["zh", "en"]

    # ── 懒加载 ────────────────────────────────────────────────────────────
    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            from my_sdk import MyModel            # 重依赖放函数内
            self._model = MyModel(self.config.engine.myengine.model_dir)
        except Exception as exc:                  # noqa: BLE001
            raise EngineError(f"加载失败 / load failed: {exc}") from exc
        self.loaded = True

    # ── 唯一的必实现方法 ──────────────────────────────────────────────────
    def transcribe(self, chunk: AudioChunk, *, language: str = "auto",
                   word_timestamps: bool = True, **kwargs: Any) -> Transcript:
        if word_timestamps:
            self.require(Capability.WORD_TIMESTAMPS,
                         "可设 transcribe.word_timestamps=false 退到句级")
        self._ensure()

        # chunk.samples 是 float32 单声道，chunk.sample_rate 通常 16000
        raw = self._model.run(chunk.samples, lang=None if language == "auto" else language)

        segments = [
            Segment(id=i, start=float(s.start), end=float(s.end), text=s.text.strip())
            for i, s in enumerate(raw.segments) if s.text.strip()
        ]
        return Transcript(
            text="".join(s.text for s in segments),
            language=getattr(raw, "language", "") or "",
            duration=chunk.duration,
            segments=segments,
            engine=self.name,
        )

    def release(self) -> None:
        self._model = None
        self.loaded = False
```

### ② 登记到 `agentic_asr/core/registry.py`

```python
ENGINES: dict[str, str] = {
    "faster_whisper": "agentic_asr.engines.faster_whisper:FasterWhisperEngine",
    # ...
    "myengine": "agentic_asr.engines.myengine:MyEngine",     # ← 加这一行
}
```

**表里存的是 `模块:类名` 字符串，不是类本身** —— 这是刻意的：列一遍引擎名不该把
faster-whisper、ctranslate2、sherpa-onnx 全 import 进来（那要几秒还占内存）。

### ③ 配置里能用它

若引擎需要自己的配置项，在 `agentic_asr/core/config.py` 里加一个 `BaseModel` 并挂到
`EngineConfig`；若只读现成字段，用 `getattr` 兜底也可以：

```yaml
# configs/asr.yaml
engine:
  default: myengine          # 或者用 --engine myengine 临时指定
  myengine:
    model_dir: models/myengine
```

命令行与 HTTP 都无需改动 —— 它们都从注册表取引擎名：

```bash
asr engines                           # 应该能看到 myengine
asr transcribe demo.mp4 --engine myengine
```

### ④ 写测试

最少两条，放进 `tests/`：

```python
def test_myengine_capabilities_without_weights() -> None:
    """能力查询不得加载权重（registry 的通用用例也会覆盖到它）。"""
    from agentic_asr.engines.myengine import MyEngine
    caps = MyEngine.declared_capabilities()
    assert Capability.TIMESTAMPS in caps


@pytest.mark.real                     # 需要真实权重的用例必须打这个 marker
def test_myengine_transcribes(tmp_path) -> None:
    ...
```

`tests/test_module.py::test_registry_lists_all_engines` 与
`test_capabilities_answerable_without_weights` 会自动把新引擎纳入检查，
**不用改它们**。

---

## 3 · 五条硬约束（每条都踩过）

### 3.1 引擎只接受 `AudioChunk`，永不接受文件路径

门面负责把任意媒体解码归一（16 kHz 单声道 float32）后交给引擎。

**理由**：faster-whisper 1.2.1 与 `av` 19 不兼容 —— 把文件路径交给它会崩在
`av.open()` 的 `metadata_errors` 参数上（`issues/001`）。根本原因则是四条输入通路
（文件 / 视频 / 字节流 / 麦克风）本就该归一成同一份表示，否则每条通路都要各自处理
重采样、降混与容器差异。

### 3.2 能力声明必须**不加载权重**就能回答

GUI 与 HTTP 在开机时用 `declared_capabilities()` 把不支持的开关置灰。
不可能为了查能力先把几百 MB 权重载进来 —— 那会让界面卡好几秒。

所以：能力写在 `classmethod` 里、用静态常量判断；**不要**在里面 `self._ensure()`。

### 3.3 不支持的能力要 `require()` 报错，绝不静默降级

```python
if word_timestamps:
    self.require(Capability.WORD_TIMESTAMPS)
```

**理由**：静默降级（比如用户要词级、你悄悄给句级）会让上层拿到一份"看起来正常、
实际缺字段"的结果 —— 下游字幕会出现无法解释的错位，而错误现场在引擎内部，
排查成本极高。宁可当场抛 `CapabilityError`，并在提示里告诉用户怎么退到句级。

门面自己也是这么做的：`ASRModule.transcribe()` 在 `want_words` 时会先 `require()`。

### 3.4 时间戳必须是**相对该 chunk** 的秒数

门面会切片（云 API 有 500 s 上限、SenseVoice 30 s 一段以拿到逐段情绪），
然后 `_shift(part, piece.start)` 平移、`_clamp(part, piece.end)` 裁剪。

**理由**：如果引擎返回的是"相对整个文件"的时间，而门面又平移一次，长音频的字幕会
整体错位 —— 且不会报错。同理，如果引擎返回超出本片长度的时间戳（Whisper 在 30 s
解码窗口里就会这样），必须靠门面的 `_clamp` 收回来。

**引擎只需保证：`0 <= start < end <= chunk.duration`。**

### 3.5 权重懒加载，构造实例必须瞬时

`__init__` 里只存配置，把加载推迟到第一次 `transcribe()`（见 `_ensure()`）。

**理由**：`ASRModule.engines()`、GUI 开机、HTTP 的 `/engines` 都会遍历引擎；
如果构造就加载权重，列个引擎名要等好几秒。另外 `engine.max_resident: 1` 下门面会
频繁装卸（8 GB 卡装不下两份），构造廉价才轮动得起。

---

## 4 · 可选实现：情绪补齐

若你的引擎能打情绪标签，实现 `detect_emotion(chunk) -> str | None` 并声明
`Capability.EMOTION`。门面在**主力引擎不支持情绪**时，会按 segment 的音频区间逐段
调用它补齐 —— 这就是「组合拳」，用户不必关心谁是主力。

参考实现：`engines/sensevoice.py`（它从 sherpa-onnx 返回的 `<|NEUTRAL|>` 标记里解析，
解析不出来时返回 `None` 而**不编造**）。

---

## 5 · 云引擎的额外注意

接云 API 时除了上面五条，还要处理：

| 事项 | 做法 | 参考 |
|---|---|---|
| 音频要编码成容器字节 | `soundfile.write(BytesIO, samples, sr, format="WAV")`，别送裸 PCM | `engines/minimax.py::_encode_wav` |
| 单次大小/时长上限 | 填 `max_audio_seconds`，让门面切片；字节数超限时抛 `APIError` | `MiniMaxConfig.max_bytes` |
| 错误映射 | 4xx/5xx → `APIError(status=..., retryable=...)`；429/5xx 标 retryable | `_STATUS_HINT` 表 |
| 厂商能力差异 | **按 (厂商, 端点, 模型) 声明能力，不按厂商硬编码** | `engines/openai_compat.py` |
| 密钥 | 只从本项目 `.env` 读（`config.api_key(engine)`），**日志里打码** | `core/config.py::Secrets` |
| 时间戳粒度 | 字/词级响应要聚合成句级 `Segment`，细粒度留在 `words` | `minimax.py::_group_units` |

最后一条值得展开：`timestamp_level=word` 时厂商返回的是**字/词级**单元，
而门面与字幕渲染都依赖句级结构。`minimax.py` 的做法是按
「说话人变化 + 时间间隙 > 0.6 s + 累计字数超限」聚合成句，同时把原始单元留在
`Segment.words` 里 —— 不要直接把字级单元当 segment 返回。
