# 006 · NiceGUI 3.x 的上传事件没有 `name` 属性

**状态**：已修复（`agentic_asr/gui/app.py::save_upload`）
**发现**：用户实测，2026-10-04

---

## 现象 / Symptom

在 GUI 里上传素材时页面报错，服务端 traceback：

```
'UploadEventArguments' object has no attribute 'name'
Traceback (most recent call last):
  File "nicegui/events.py", line 495, in _await_and_handle_in_context
    await awaitable
  File "agentic_asr/gui/app.py", line 66, in on_upload
    dest = cfg.output_dir / "uploads" / (e.name or "upload.bin")
AttributeError: 'UploadEventArguments' object has no attribute 'name'
```

上传**完全不可用**（界面不崩，但文件没存下来，路径框也不会被填上）。

## 判定 / Diagnosis

代码是照 **NiceGUI 2.x** 的 API 写的（`e.name` + `e.content.read()`），
但这台机器上是 **NiceGUI 3.16.0**，事件对象结构已变：

```python
# NiceGUI 3.x
@dataclass(kw_only=True, slots=True)
class UploadEventArguments(UiEventArguments):
    file: FileUpload        # 只有这一个字段
```

而 `FileUpload`（`nicegui/elements/upload_files.py`）是一个 ABC：

| 成员 | 说明 |
|---|---|
| `name: str` | 文件名（**在 file 上，不在事件上**） |
| `content_type: str` | MIME |
| `await save(path)` | **直接把内容写到路径**（推荐用法） |
| `await read() -> bytes` | 读成 bytes |
| `size() -> int` | 大小 |

⇒ 2.x 的 `e.name` / `e.content` 在 3.x 里都不存在。

## 修法 / Fix

抽出 `save_upload(file, dest_dir)`，用 3.x 的契约：

```python
name = Path(str(getattr(file, "name", "") or "upload.bin")).name   # 只取 basename
dest = Path(dest_dir) / name
dest.parent.mkdir(parents=True, exist_ok=True)
await file.save(str(dest))
```

三点考虑：

1. **只取 basename**：客户端可以在 `name` 里塞 `../../evil.wav`，
   不处理就是目录穿越；
2. **用 `await file.save()`** 而不是 `read()` + 自己写 —— 框架的实现会分块流式落盘，
   大文件不必整个进内存；
3. **包一层 try/except**：上传失败只改状态栏，不让整个界面挂掉。

同一个写法在**另一个模组的 GUI** 里也存在（两边各自修各自的，互不引用）。

## 回归用例 / Regression

`tests/test_gui.py`（两个仓库各一份）：

- `test_save_upload_writes_file` —— 走通 3.x 的落盘路径；
- `test_save_upload_strips_client_path` —— `../../evil.wav` 必须被削成 `evil.wav`；
- `test_save_upload_handles_missing_name` —— 空名字回落到 `upload.bin`；
- `test_upload_event_arguments_shape` —— **把框架契约本身钉住**：
  断言 `file` 在事件字段里、而 **`name` / `content` 不在**。
  将来 NiceGUI 再改 API 会先撞到这条，而不是等用户上传时才炸。

## 备注

上传之外，GUI 的其余部分是真实可用的：两个界面都已实际构建验证通过。
真正需要浏览器的交互仍属人工验证范围（见 `docs/07_gui_guide.md`）。
