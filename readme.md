## 项目简介

SGS-PZ 是一个服务端判定的「三国杀（身份局）」规则引擎，提供：
- PettingZoo AEC 多智能体环境（适配 Gymnasium 终止/截断语义）
- 固定动作空间 + 合法动作掩码
- FastAPI 网关（HTTP + WebSocket）与简易 Web UI
- RLDS 训练日志（含 DPO/GRPO 数据），可直接喂给 RL/TRL 训练
- 可插拔 LLM 适配器（OpenAI / OpenRouter / 自定义 HTTP）

该项目支持人类/LLM/Bot 混合对局与高并发自博弈数据生成，便于端到端训练与评测。

## 已实现能力

- 规则与流程
  - 回合六阶段：准备/判定/摸牌/出牌/弃牌/结束
  - 距离与攻击范围：环座距离，±1 马，武器射程
  - 响应链/事件栈：杀-闪、无懈栈（嵌套奇偶取消）、决斗、群体锦囊（南蛮/万箭）
  - 延时判定：乐/兵粮/闪电（判定顺序、花色/点数判定、唯一性与传递）
  - 濒死窗口：先救治（桃）后死亡，继而身份奖惩与胜负判定
  - 身份胜负：主/忠/反/内，击杀奖惩（反贼赏三、误杀忠臣主公弃牌）与终局判断
- 武将（10）与技能（常见基础版）：
  - 张飞（咆哮）、关羽（武圣）、赵云（龙胆）、马超（铁骑）、诸葛亮（空城）
  - 曹操（奸雄）、司马懿（反馈）、夏侯惇（刚烈）、孙权（制衡）、华佗（青囊）
- 装备与效果（示例）：
  - 武器：诸葛连弩（无限杀）、青釭（无视防具）、青龙（追加杀）、丈八（两牌当杀）、方天、麒麟（拆坐骑）、贯石（闪后强伤）等
  - 防具：八卦（红判当闪）、仁王（黑杀无效）、藤甲/白银（简化处理）
  - 坐骑：±1 马
- 牌堆与花色/点数
  - 接入 data/standard_deck_108.json 为牌源（当前覆盖已实现牌种；其余牌种将逐步补齐）
  - 全面使用花色/点数参与判定与若干技能/防具交互
- 强化学习与日志
  - PettingZoo AEC 环境（合法掩码在 info["legal_action_mask"]）
  - RLDS 步进日志（JSONL），并额外输出 DPO 成对样本与 GRPO 候选组
  - 头less 并发自博弈（线程池），支持多局并行生成数据
- LLM 适配
  - OpenAI/OpenRouter/自定义 HTTP 端点，超时/异常自动降级随机合法动作
  - Prompt 包含 observation、近 32 步历史简要（agent/phase/events/action）与合法动作索引列表
- 网关与 UI
  - FastAPI：REST（建房/加入/并发运行/元信息）+ WS（对局流）
  - Web UI：创建房间、观战/控制座位、动作掩码点击、人手牌/装备/判定区、公开信息（所有玩家 hp/手牌数/装备范围/判定区数）、可选目标提示、事件时间线

## 目录结构

- `sgs_env/`：环境与规则（状态、观测、掩码、步进、结算）
- `agents/`：随机 Bot、LLMAdapter
- `coordinator/`：本地单局与并发自博弈协调器
- `recorder/`：RLDS 记录器（含 DPO/GRPO）
- `api/`：FastAPI 应用与静态 Web UI
- `data/`：标准 108 张牌表（JSON）
- `tests/`：规则/组合/API/WS 冒烟测试

## 安装与测试

- 安装（开发模式）：
  - `pip install -e .[dev]`
- 运行测试：
  - `pytest`（如无全局 pytest，用 `~/.local/bin/pytest`）

## 命令行（CLI）

安装后提供 `sgs-cli`：
- 本地交互游玩（你控制座位 0，其他为随机 Bot）：
  - `sgs-cli play --seed 0 --num-players 4 --max-steps 200`
- 批量自博弈并生成日志：
  - `sgs-cli selfplay --seed 0 --num-players 4 --episodes 16 --parallelism 8 --max-steps 200 --record true`
- 启动 API 服务：
  - `sgs-cli api --host 0.0.0.0 --port 8000`

## Web API 与 WebSocket

- 启动服务：
  - `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- 打开 Web UI：
  - 浏览器访问 `http://localhost:8000/`
  - 创建房间 → 复制 `game_id` → 选择 `seat`（-1 观战，0..N 控制）→ 连接/观战
- REST 接口：
  - POST `/rooms`：创建房间，返回 `{game_id}`
  - POST `/rooms/join`：加入座位（可选 human/llm/bot；llm 可指定 provider/model）
  - POST `/headless/run`：批量自博弈，参数 `seed/num_players/record/num_episodes/parallelism/max_steps`
  - GET `/meta/cards`：返回 `card_id -> 牌名` 映射
- WebSocket 接口：
  - `/ws/game/{game_id}?seat=<n>`：连接对局流；`seat>=0` 时该连接控制此座位，否则为观战
  - 服务器每步推送：
    - `agent` 当前出手者
    - `info.mask` 合法动作掩码
    - `info.phase` 阶段，`info.events` 事件列表
    - `info.observation` 当前可观测结构（含 `self/public/phase/turn_agent` 等）
    - `info.seating` 座次（agent/seat/alive），`info.targets` 可选目标（含各目标位对应的动作索引）
  - 客户端发送：`{"action": <int>}` 选择一个合法动作索引

## LLM 集成

- 环境变量（任选）：
  - `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`
  - `OPENROUTER_API_KEY`、`OPENROUTER_BASE_URL`、`OPENROUTER_MODEL`
  - 自定义：`LLM_PROVIDER`、`LLM_MODEL`、`LLM_ENDPOINT`
- LLMAdapter.act(observation, legal_mask, history) 行为：
  - 构造安全提示 + 传入 observation 与最近 32 步历史摘要
  - 解析模型输出中的第一个整数，若非法或超时/异常则降级为随机合法动作

## RLDS 数据

- 生成位置：`replays/YYYYMMDD/*.rlds.jsonl`
- 步进记录字段（StepRecord）：`episode_id/step/agent/observation/action/reward/terminated/truncated/info`
- DPO 样本：`replays/YYYYMMDD/dpo_pairs.jsonl`（每步从合法集中采一个 rejected）
- GRPO 组：`replays/YYYYMMDD/grpo_groups.jsonl`（每步截取至多 4 个候选）
- 与 TRL 对接：
  - 可编写 loader 读取 JSONL 并转为 HF Datasets，字段中 `action` 即监督信号，或将 DPO/GRPO 文件直接转化为 TRL 接口所需结构

## 动作空间（概览）

- 固定大小（见 `sgs_env/constants.py` 与 `sgs_env/actions.py`）：
  - PASS/CONFIRM/TAO/响应（闪/杀/无懈）
  - 杀（按座位索引映射）、决斗/过拆/顺手/乐/兵粮（按目标座位映射）
  - 群体锦囊：南蛮/万箭
  - 延时：闪电（置于自身判定区）
  - 装备：武器/防具（含坐骑）
  - 弃牌槽位（制衡阶段/弃牌阶段使用）

## 开发与测试

- 测试集包含：
  - 基础环境/战斗/英雄与装备交互/无懈栈/主动技/规则组合/整局冒烟
  - API/WS 冒烟（观战/座位控制、元信息接口）
- 运行：`pytest`

## 当前限制与后续计划

- 108 张牌堆当前仅包含已实现牌面；未实现的基础牌（如 酒/无中/五谷/借刀/铁索/火攻/桃园/火杀/雷杀 等）将与规则/测试一并补齐后替换为完整官方表
- 部分装备与技能为精简实现，将逐步完善其边界与联动
- Web UI 目前为简洁版本，后续将完善：多客户端座位占用策略、更多公开信息细化、日志时间线细节、回放
- 提供 RLDS→TRL 的示例脚本与更完整的训练样例

## 许可证

本项目面向研究用途与工程验证，涉及到的三国杀名词/规则等版权归原版权方所有。
