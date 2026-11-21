# BTC 24H 极值磁吸带 & Deribit GEX 实时监控看板 (V4 全自动 ETF 净流入版)

## V4 核心升级
1. **美国现货 BTC ETF 净流入双层全自动抓取（30 分钟缓存）**：
   - **第一优先级（推荐）**：接入 **SoSoValue 官方开放 API**（`GET /openapi/v1/etfs/summary-history?symbol=BTC&country_code=US`，免费每月 10,000 次无需绑卡）。自动跳过 T+1 尚未结算的 `null` 行，精准获取最新已结算交易日数据。
   - **第二优先级（零密钥开箱即用）**：若未配置 `SOSO_API_KEY`，自动解析 **Farside Investors** 官方日度流量表 (`https://farside.co.uk/btc/`)，精准识别括号负值（如 `(124.5)`）与最新有效交易日。
   - **自动联动事件冲击模型**：当最新单日净流入/流出绝对值达到 `±$100M` 以上时，自动注入系统情绪与四象限路由引擎。
2. **网页端新增 ETF 实时 KPI 卡片**：展示最新单日净流入金额、所属交易日与数据来源；底部覆盖框留空即自动恢复全自动抓取。

## 快速启动
- **Windows**：解压后双击 `start_dashboard.bat` 即可运行。
- **macOS / Linux**：执行 `chmod +x start_dashboard.sh && ./start_dashboard.sh`。
- **开启 100% 稳定官方 ETF API（可选，1 分钟搞定）**：访问 `https://sosovalue.com/developer/console/keys` 免费生成 Key，复制到 `.env` 中的 `SOSO_API_KEY=你的key` 即可。
