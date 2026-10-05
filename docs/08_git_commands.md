# 待执行的提交命令

> **这个文件是一次性的。** 每次覆写，只留当前这一组；上一组的内容在 `git log` 里。
>
> **覆写前必须先 `git log` 核对上一组是否已执行。** 本次核对结果（2026-10-05）：
> 上一组「字幕优化与 lint 清理」**已全部执行**（`88f0258` → `d9464a3` 之后的那批）。
> 故本次覆写是安全的；当前这一组是**模组解耦 + 字幕软/硬上限 + lint 清理**，共 4 个 commit。
>
> `git add` 一律写明路径，**不用 `-A`**：`.env` 有密钥、`models/` 有权重、
> `outputs/` 有几百 MB 产物，它们由 `.gitignore` 挡住。
> hook 会拦下 `git commit` / `git push` —— 这些命令由人工执行。

**提交前核对**：

```bash
git status --short              # 只应有本组列出的那些文件
python tools/check.py           # 唯一的闸：ruff + 全量测试 + 环境自检
```

commit 类型前缀用英文，正文用中文。

---

## ① 模组解耦：跨模组的脚本移出，文档不再直接引用

```bash
git reset
git add README.md agentic_asr/core/config.py \
  docs/00_INDEX.md docs/00_STAGE_SUMMARY.md docs/00_research.md docs/01_design.md \
  docs/02_dev_plan.md docs/03_unit_tests.md docs/05_deployment.md docs/09_dev_log.md \
  docs/PROJECT_MEMORY.md docs/issues/006-nicegui-3-upload-event-has-no-name.md \
  scripts/_smoke_pipeline.py scripts/_probe_separate.py \
  scripts/_probe_separation_sherpa.py scripts/_probe_leadvocal.py
git commit -m "refactor: 与声音分离模组解耦，跨模组的脚本移出或删除

按守则「模组之间不许互相指引」清理。三种形态各自处理：

- **跨模组脚本移出**：scripts/_smoke_pipeline.py 同时 import 两个模组，
  它属于**应用层**，已移到 F:\\2026年\\Agentic_Pipelines\\（连同它要用的
  样例素材，不再从本仓库的 models/ 里取）
- **放错位置的探针删除**：scripts/ 下三个**属于分离模组**的调研探针
  （demucs 可行性、sherpa UVR 实测、主体/背景人声区分）——
  它们的调研任务已完成，结论分别落盘在 00_research.md §4.2/§4.4 与
  分离模组自己的文档里，脚本本身没有保留价值
- **文档直接引用**：删掉 01_design.md 里整节「分离插件设计」（引擎抽象、
  模型清单、轨道判定、等长约束 —— 那是**另一个模组的设计正文**）、
  目录结构里的 agentic_separate/、API 示例与端口表里的分离条目
- **注释示例**：core/config.py 里那段可直接照抄的 SeparateModule() 调用删掉，
  改成说明「由应用层组合」

保留的是**关系说明**（不指名道姓、不给 API）：两者能力正交、互不依赖，
要一起用由应用层组合、脚本放模组之外。跨模组的事实只在它自己的仓库写权威。

顺带清掉几处已过时的自述：不再自称「两个插件」，configs 不再列 separate.yaml，
.env 不再提 SEPARATE_SERVER_PORT。"
```

## ② 字幕：字数上限从「硬切」改成软/硬两级

```bash
git reset
git add agentic_asr/subtitle/segment.py configs/asr.yaml tests/test_subtitle.py \
  docs/issues/007-asr-hides-silence-in-a-single-unit.md
git commit -m "fix(subtitle): 字数上限改成软/硬两级，不再切在词中间

- 原判据是 buf_len >= max_chars 就切，不看下一个字符是不是标点 ——
  中文没有词边界，实测切出过「函 / 数」「基 / 本」
- 现在到 max_chars 只表示「该切了」，还要继续往前走到下一个标点；
  超过 max_chars × hard_chars_ratio（默认 1.6，即 32 字）仍无标点才无条件切
- 两处都改：segment_units()（走字级时间戳的主路径）与
  _split_text_chunks()（无时间戳的回退路径，此前同样是硬切）
- hard_chars_ratio 进 SegmentParams 与 configs/asr.yaml，可调
- 01_design §3.5 的规则表从六条改为七条；issues/007 补记这一半问题
- 新增三条回归用例：软上限等到标点 / 无标点时短串保留 / 超过硬上限必切"
```

## ③ lint 清理（ruff 第一次跑全仓发现的历史遗留）

```bash
git reset
git add pyproject.toml tools/check.py agentic_asr/core/types.py \
  agentic_asr/engines/base.py agentic_asr/engines/faster_whisper.py \
  agentic_asr/engines/openai_compat.py agentic_asr/asr.py agentic_asr/audio/io.py \
  agentic_asr/core/registry.py agentic_asr/engines/minimax.py agentic_asr/engines/mock.py \
  agentic_asr/gui/app.py agentic_asr/media/probe.py \
  tests/test_guard.py tests/test_module.py tests/test_server_cli.py
git commit -m "chore: 清掉 ruff 的历史遗留告警

- pyproject.toml 补 ruff 配置：typer 与 FastAPI 的 Option/File/Form 进
  extend-immutable-calls —— 那是两个框架的标准写法，B008 要防的不是它们，
  B027 的 ASREngine.release 同样是「可选钩子」而非漏写的 abstractmethod
- tools/check.py 删掉重复的 Python 版本判断（requires-python 已保证）
- 真 bug 只有三处：未使用的 import（死代码）
- 其余是风格与现代化：import 排序、typing.Callable → collections.abc、
  f-string 无占位符、未使用的循环变量
- Capability / Emotion 改用 StrEnum：**取字符串时行为有变**，
  现在 str(Capability.EMOTION) 得到 \"emotion\" 而不是 \"Capability.EMOTION\"；
  已确认没有调用方依赖旧写法"
```

## ④ 清单自身

```bash
git reset
git add docs/08_git_commands.md
git commit -m "docs: 提交清单推进到模组解耦与字幕上限"
```

---

## 核对

```bash
git status --short      # 应当干净
git log --oneline -6
```
