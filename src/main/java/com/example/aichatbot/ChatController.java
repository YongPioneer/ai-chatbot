package com.example.aichatbot;

import java.nio.charset.StandardCharsets;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.core.io.ClassPathResource;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ChatController {

    private final ChatClient chatClient;

    public ChatController(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    @GetMapping("/chat")
    public String chat(@RequestParam String msg) {
        return chatClient.prompt(msg).call().content();
    }

    // 外贸销售助手：角色Prompt + 产品资料文件
    @GetMapping("/sales")
    public String sales(@RequestParam String msg) {
        String systemPrompt = """
                你是一名专业的外贸销售，代表一家中国电气元器件工厂，主营接触器(contactors)、继电器(relays)、隔离开关(disconnect switches)、浪涌保护器(surge protectors)，产品通过UL认证。
                你的职责是处理海外客户的询盘，回答产品规格、MOQ、交期、认证、付款方式等问题，目标是专业、得体地推动客户进入下一步沟通。
                回复要求：
                1. 用英文回复，语气专业、友好、谦逊
                2. 回答简洁清晰，关键信息用要点列出
                3. 优先依据下方"公司产品资料"回答；资料中没有的信息，说明需要与工厂核实，建议客户联系销售获取详细报价
                4. 结尾适当引导客户提供更多需求细节（如目标市场、预估数量、应用场景、是否需样品）
                """ + "\n\n【公司产品资料】\n" + loadProducts();
        return chatClient.prompt()
                .system(systemPrompt)
                .user(msg)
                .call()
                .content();
    }

    // 从 resources/products.txt 读取产品资料
    private String loadProducts() {
        try {
            ClassPathResource resource = new ClassPathResource("products.txt");
            return new String(resource.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "（产品资料暂未加载）";
        }
    }
}
