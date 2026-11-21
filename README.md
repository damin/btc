# BTC 24H 极值磁吸带 & Deribit GEX 实时监控看板 (V3 美国直连纯净版)

## V3 架构精简与升级
- **彻底移除受限数据源**：零冗余代码、零无效重试延迟，美国本土本地电脑及美东云服务器均免代理直连。
- **主数据源：OKX V5 公开永续合约 (`BTC-USDT-SWAP`)**：实时提供现价、48H 1H K 线（锚定 AVWAP 与精确已收盘 1H K 线）、±1% 盘口深度、8H 资金费率及 1H/24H OI 变化率。
- **自动备援源：Deribit 公开永续合约 (`BTC-PERPETUAL`)**：与期权数据源同源复用，当 OKX 偶发超时或维护时，15 分钟熔断冷却期内自动无缝切换。
- **期权与波动率源：Deribit 公开 API**：实时计算真实 DVOL、全行权价 Net GEX 分布、Gamma Flip、Call Wall 与 Put Wall。

## 快速启动
- **Windows**：解压后双击 `start_dashboard.bat`，自动建环境并打开 `http://127.0.0.1:5050`。
- **macOS / Linux**：终端执行 `chmod +x start_dashboard.sh && ./start_dashboard.sh`。
- **云端部署 (Render.com)**：推送至 GitHub 后通过 `New + → Blueprint` 导入 `render.yaml`（已预设美东 `virginia` 节点）。
