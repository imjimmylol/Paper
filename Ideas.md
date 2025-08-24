# A. Multi-agnet (Advanced)

## AlphaStar 演算法概覽
- <mark>function-arguments 自回歸編碼 可用於經濟問題 壓縮agent的活動空間</mark>
- <mark>參數共享的經濟意義：agnet 處理shock the common reaction，再到個體的差異部分</mark>
- <mark>行為模仿可以對標現實世界經濟資料(但是資料量可能很不足)</mark>


| 階段 | 技術 / 做法 | 目的 |
|------|-------------|------|
| 1. 模擬介面 | *SC2LE* API；動作以 **function-arguments** 自回歸編碼 | 壓縮 & 合法化動作空間 |
| 2. 神經網路 | 區域卷積 + *Relational* 注意力 → LSTM；同時輸出 π(a \| s) 與 V(s) | 建模多單位關係與長程依賴 |
| 3. 行為模仿 (BC) | 約 50 萬局人類重播上行為克隆 | 提供可玩、接近人類的起點 |
| 4. RL 微調 | IMPALA / **V-trace** off-policy actor-critic；勝負 (+1/0) 回饋 | 穩定且可分散更新 |
| 5. League-based Self-Play | 角色：主力、主力剋星、聯盟剋星… 交叉對戰並共享回放 | 規避循環最適、增加策略多樣性 |
| 6. Population-Based Training (PBT) | 週期性複製高勝率權重並隨機微調超參 | 自動搜尋學習率、熵懲罰等 |
| 7. 延遲 / 操作限制 | 模擬人類 350 APM / 60 fps 反應上限 | 公平比較，避免超人反應 |

---

## 關鍵訓練流程

1. **Bootstrap**：以 BC 權重初始化並將重播存入回放池  
2. **分散資料收集**：數千 Actor 自我對戰寫回放  
3. **Learner 更新**：GPU/TPU 伺服器用 V-trace 校正的策略梯度 + TD(λ) 更新  
4. **League 排程**：依 Elo 與角色規則指派對手  
5. **PBT 週期**：每 ~200k 梯度步根據勝率複製 / 探索成員  

---

## 成果

- 2019 年版本三族皆達 **Grandmaster**（> 99.8% 玩家分位）  
- 證明「離線模仿 + 大規模對抗式 RL + 自適應超參」可處理高維、部分可觀測、長時序之即時戰略環境  

---

## 主要文獻

- Vinyals et al., *Grandmaster level in StarCraft II using multi-agent reinforcement learning*, **Nature**, 2019  
- Mathieu et al., *AlphaStar Unplugged: Large-Scale Offline RL*, arXiv:2308.03526, 2023



# B. Simple Start : Single Agnet 

## 0. Issue 
### Backbone : Monetary Policy and Exchange Rate Volatility in a Small Open Economy 



## 1. Constarint problem 

### RL 中的 Agent 與 Environment 限制機制

| 層級 | 約束機制 | 說明 / 常見做法 |
|------|----------|----------------|
| **介面層 (API)** | **Action Space 定義**<br>– Discrete：列舉有限動作集合<br>– Continuous：箱形 / 多邊形區間，常見 `clip()`、`tanh` 壓縮 | 超出範圍的動作直接截斷或拒收 |
| **狀態依賴合法性** | **Action Mask / Legal-Action Set**<br>如棋類、SC2 function-ID，每步根據 state 提供布林遮罩 | Policy 取樣前 `mask_logits`；非法動作機率設 −∞ |
| **環境動力學** | **物理／邏輯約束**<br>作用力上限、關節角度、庫存數量、邊界牆 | `env.step(a)` 判斷；違規時 ① 投影 ② 罰分 ③ 終止 |
| **安全層 / Shield** | 行為前置 **Safety Filter** / CBF；覆寫或選擇最近安全動作 | 常用於機器人、交通；確保碰撞距離、電流極限 |
| **優化層 (算法)** | **Constrained RL / CMDP**<br>成本 `c(s,a)` + Lagrangian、Primal-Dual、PID-λ 等 | 滿足期望成本 ≤ η 的軟約束 |
| **輸入正則化** | 梯度懲罰、熵獎勵、KL 節制 | 避免策略逼近邊界梯度爆炸或過度隨機 |
| **基礎控制器** | **階層式 RL**：高層輸出抽象動作，低層由硬體控制器 / MPC 保證可行 | 高層規劃，低層確保物理合法 |

---

### Environment Constraints

- **狀態空間邊界**：位置、速度、存款額度等自然上限  
- **轉移規則**：`s_{t+1} = f(s_t, a_t)` 內建守恆、碰撞、失敗模式  
- **終止條件**：碰撞、越界、違規即 `done = True`  
- **隱藏資訊 / POMDP**：限制觀測而非動作，間接促進安全行為  

---

### 實務摘要

1. **硬約束**：在 action space / dynamics 層面直接保證合法性。  
2. **狀態依賴約束**：使用 Mask 或 Shield，保持梯度可傳遞。  
3. **軟約束**：Constrained RL 控制期望違規次數或能量消耗。  
4. **分層控制**：高層專注規劃，低層確保物理可行。  


## 2. Agent Settings
把planner 換成 Transformer, 這樣可以處理時序不一致，目標可以極小化 政策過程中的約束