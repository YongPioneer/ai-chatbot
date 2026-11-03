# AI外贸销售助手（ai-chatbot）

基于 **Spring Boot 4 + Spring AI 2.0** 构建的中英双语 RAG 智能销售助手，面向电气元器件（接触器 / 继电器 / 隔离开关 / 浪涌保护器）外贸 B2B 场景。客户询盘进来 → 从产品资料库检索 → DeepSeek 组织回答，全程支持 SSE 流式输出。

## 功能亮点

- **RAG 双通道检索**：关键词精确匹配（型号归一化，`SA-3P ≡ SA 3P ≡ SA3P`）+ 向量语义检索（top8），型号类问题不丢、语义类问题能补
- **本地向量化零成本**：ONNX 多语言模型（paraphrase-multilingual-MiniLM-L12-v2，384 维）进程内直载，不依赖任何云端 embedding 服务
- **向量库 PostgreSQL + pgvector**：2979 段产品资料持久化入库，余弦距离检索，重启秒启（8s），支持 HNSW 索引扩展
- **中英双语**：中文问中文答、英文询盘英文答，多语言向量模型 + 中文产品资料天然支持
- **SSE 流式**：`/rag-stream` 打字机效果，首字 1 秒内可见
- **Agent 工具调用（Function Calling）**：`/agent` 接口让 DeepSeek 自主决定调用 `queryProduct` / `findProductsByNeed` 工具查资料，支持报型号、报规格、英文询盘
- **资料热插拔**：`docs/` 放 PDF/Word/TXT/MD，重启自动增量入库（文件指纹驱动缓存/重建）

## 技术栈

| 层 | 技术 |
|---|---|
| 框架 | Spring Boot 4.1.2 / Spring AI 2.0.0 / Java 21 |
| 推理 | DeepSeek（火山方舟 API） |
| 向量化 | TransformersEmbeddingModel（ONNX 多语言 MiniLM，384 维） |
| 向量库 | PostgreSQL 17 + pgvector 0.8.7（COSINE_DISTANCE） |
| Agent | Spring AI @Tool 函数调用（queryProduct / findProductsByNeed） |
| PDF 解析 | PyMuPDF（`scripts/extract_pdf_pages.py`，失败回退 PDFBox） |
| 输出 | REST `/rag` + SSE `/rag-stream` + Agent `/agent` |

## 架构

```
客户询盘（中/英）
   │
   ▼
┌───────────── RAGController ─────────────┐
│ ① 关键词通道：型号 token 精确匹配（归一化）│
│ ② 系列通道：系列前缀跨文件取样             │
│ ③ 向量通道：pgvector 余弦 top8            │
│  → 合并去重取 4 段                        │
└────────────────┬────────────────────────┘
                 ▼
        DeepSeek（火山方舟）组织回答
                 ▼
        外贸销售语气 + 参数 + 引导话术
```

## 快速开始

### 环境要求

- JDK 21、Maven（项目自带 mvnw）
- PostgreSQL 17 + pgvector 扩展（`CREATE EXTENSION vector;`，建库 `ai_chatbot`）

### 配置环境变量（密钥不落盘）

| 变量 | 说明 | 默认 |
|---|---|---|
| `ARK_API_KEY` | 火山方舟 API Key（**必填**，无默认值） | — |
| `DB_PASSWORD` | 数据库密码 | `postgres` |
| `DB_USER` | 数据库用户名 | `postgres` |

命令行设置：

```powershell
$env:ARK_API_KEY="你的火山方舟Key"
```

IDEA 中设置：`Run/Debug Configurations → Environment variables` 填 `ARK_API_KEY=xxx`。

### 启动

```powershell
.\mvnw.cmd spring-boot:run
```

首次启动自动：扫描 `docs/` → PyMuPDF 提取 → 按行切块 → 向量化 → 入库 PostgreSQL（约 2-4 分钟，日志见"构建完成"）。

### 试用

```bash
# 普通问答
curl "http://localhost:8080/rag?msg=PR360%E7%9A%84%E5%8F%82%E6%95%B0"

# 流式（SSE 打字机）
curl -N "http://localhost:8080/rag-stream?msg=send%20me%20the%20parameters%20of%20SA-3P"

# Agent（模型自主调用工具查资料）
curl "http://localhost:8080/agent?msg=PR360%E7%9A%84%E5%8F%82%E6%95%B0"
```

### 喂新资料

把 PDF/Word/TXT/MD 放进 `docs/` 目录 → 重启应用 → 自动增量入库（无需任何操作）。

## 目录结构

```
ai-chatbot/
├─ src/main/java/com/example/aichatbot/
│  ├─ RAGController.java     # /rag + /rag-stream + /agent 接口
│  ├─ RAGConfig.java         # 向量模型、切块、知识库加载（缓存/重建）
│  ├─ ProductSearchService.java  # 双通道检索（RAG/Agent 共用）
│  ├─ SalesTools.java        # @Tool 工具：queryProduct / findProductsByNeed
│  └─ AiChatbotApplication.java
├─ src/main/resources/
│  ├─ application.yml        # 数据源 + 向量库 + 模型配置
│  └─ products.txt           # 内置产品速查
├─ docs/                     # 产品资料（PDF/Word/TXT/MD，热插拔）
├─ models/                   # 本地 ONNX 向量模型（118MB，git 忽略）
├─ data/                     # 关键词缓存 + 文件指纹（git 忽略，删掉重建）
└─ scripts/extract_pdf_pages.py  # PyMuPDF 逐页提取
```

## 已知边界与优化方向

- 表格列语义丢失：PDF 表格拆成裸数字行后列名丢失（下一步：表格结构化入库 `find_tables()`）
- 单轮问答：暂无多轮记忆（下一步：Spring AI ChatMemory + PG 存储）
- 无 Agent：暂不能自动写开发信/查库存（下一步：@Tool 函数调用）
- API Key 已环境变量化，可安全推 GitHub
