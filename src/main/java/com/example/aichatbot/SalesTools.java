package com.example.aichatbot;

import java.util.List;

import org.springframework.ai.document.Document;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

/**
 * 外贸销售Agent工具集（课2：Function Calling / 工具调用）
 *
 * 这些方法会暴露给大模型（DeepSeek），模型判断"需要产品资料"时自主调用，
 * 拿到返回结果再组织回答——这就是 Agent 的核心循环：模型决定 → 调工具 → 基于结果生成。
 *
 * 注意：工具方法是确定性逻辑（查库/格式化），不在这里再调LLM。
 */
@Component
public class SalesTools {

    private final ProductSearchService searchService;

    public SalesTools(ProductSearchService searchService) {
        this.searchService = searchService;
    }

    /**
     * 工具1：按型号/系列查询产品参数资料。
     * 模型会在客户询盘提到型号（如PR360、SA-3P、PR系列）时自动调用。
     */
    @Tool(description = "根据客户提到的产品型号或系列查询产品参数资料（如 PR360、SA-3P、PR系列、HCDP-2P）。"
            + "型号可以用中文、英文或混合提问，支持空格/连字符变体。返回匹配的产品资料文本，未找到返回空。")
    public String queryProduct(
            @ToolParam(description = "产品型号或系列名称，例如：PR360 / SA-3P / PR系列 / HCDP-2P") String modelOrSeries) {
        List<Document> docs = searchService.retrieve(modelOrSeries);
        if (docs.isEmpty()) {
            return "未检索到与【" + modelOrSeries + "】相关的产品资料，请告知客户需要与工厂核实。";
        }
        return searchService.buildContext(docs);
    }

    /**
     * 工具2：按客户需求（规格/用途）推荐产品。
     * 模型会在客户描述需求而不是报型号时（如"我要24V线圈SPST-NO的继电器"）调用。
     */
    @Tool(description = "根据客户的规格或用途描述（如：24V线圈、SPST-NO、用于充电桩的继电器）"
            + "在知识库中检索可能匹配的产品资料。返回相关资料文本。")
    public String findProductsByNeed(
            @ToolParam(description = "客户描述的规格/用途，例如：24V 线圈 SPST-NO 继电器 充电桩") String requirement) {
        List<Document> docs = searchService.retrieve(requirement);
        if (docs.isEmpty()) {
            return "未检索到匹配该需求的产品资料，建议引导客户提供具体型号或联系工厂定制。";
        }
        return searchService.buildContext(docs);
    }
}
