package com.example.aichatbot;

import java.util.List;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

/**
 * 外贸销售助手接口：
 *
 * /rag  + /rag-stream   RAG问答（检索→拼接→回答，资料由服务端硬塞给模型）
 * /agent               Agent问答（课2：模型自己决定何时调用 queryProduct 工具查资料）
 *
 * 课2要点：ProductSearchService 抽成独立服务，RAG 和 Agent 共用同一套检索逻辑（单一职责）；
 *         Agent 版通过 ChatClient.tools(...) 把 @Tool 方法暴露给 DeepSeek。
 */
@RestController
public class RAGController {

    private final ChatClient chatClient;
    private final ProductSearchService searchService;
    private final SalesTools salesTools;

    public RAGController(ChatClient.Builder builder, ProductSearchService searchService, SalesTools salesTools) {
        this.chatClient = builder.build();
        this.searchService = searchService;
        this.salesTools = salesTools;
    }

    private static final String SYSTEM_PROMPT = """
            你是一名专业的外贸销售，代表一家中国电气元器件工厂，主营接触器(contactors)、继电器(relays)、隔离开关(disconnect switches)、浪涌保护器(surge protectors)，产品通过UL认证。
            回答客户询盘时，必须优先依据【检索到的产品资料】回答，语气专业、友好、谦逊。
            资料中没有的信息，说明需要与工厂核实，建议客户联系销售获取详细报价。
            结尾适当引导客户提供更多需求细节（目标市场、预估数量、应用场景、是否需样品）。
            """;

    /** Agent 版系统提示：模型自主决定调用工具查资料 */
    private static final String AGENT_SYSTEM_PROMPT = """
            你是一名专业的外贸销售Agent，代表一家中国电气元器件工厂，主营接触器(contactors)、继电器(relays)、隔离开关(disconnect switches)、浪涌保护器(surge protectors)，产品通过UL认证。
            回答客户询盘前，先判断是否需要产品资料：
            - 客户提到具体型号/系列（如PR360、SA-3P、PR系列）→ 调用 queryProduct 工具查询真实资料
            - 客户描述规格/用途（如24V线圈、SPST-NO、充电桩用）→ 调用 findProductsByNeed 工具查找匹配产品
            - 客户要开发信/询盘回复 → 先查资料，再基于真实资料起草
            基于工具返回的真实资料回答，语气专业、友好、谦逊；资料中没有的信息，说明需与工厂核实。
            中文询盘用中文回答，英文询盘用英文回答。结尾适当引导客户提供目标市场、预估数量、应用场景、是否需样品。
            """;

    // ---- RAG问答（资料由服务端检索后硬塞给模型）----
    @GetMapping("/rag")
    public String rag(@RequestParam String msg) {
        System.out.println("[RAG调试] 问题: " + msg);
        List<Document> docs = searchService.retrieve(msg);
        return chatClient.prompt()
                .system(SYSTEM_PROMPT)
                .user("【检索到的产品资料】\n" + searchService.buildContext(docs) + "\n\n【客户问题】" + msg)
                .call()
                .content();
    }

    // ---- RAG流式问答（SSE打字机效果）----
    @GetMapping(value = "/rag-stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> ragStream(@RequestParam String msg) {
        System.out.println("[RAG调试] 流式问题: " + msg);
        List<Document> docs = searchService.retrieve(msg);
        return chatClient.prompt()
                .system(SYSTEM_PROMPT)
                .user("【检索到的产品资料】\n" + searchService.buildContext(docs) + "\n\n【客户问题】" + msg)
                .stream()
                .content();
    }

    // ---- Agent问答（模型自主调用工具）----
    @GetMapping("/agent")
    public String agent(@RequestParam String msg) {
        System.out.println("[Agent调试] 问题: " + msg);
        return chatClient.prompt()
                .system(AGENT_SYSTEM_PROMPT)
                .tools(salesTools)               // 暴露 @Tool 方法给 DeepSeek（Function Calling）
                .user("【客户问题】" + msg)
                .call()
                .content();
    }
}
