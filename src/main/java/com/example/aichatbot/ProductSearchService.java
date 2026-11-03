package com.example.aichatbot;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

/**
 * 产品检索服务（课2：抽取为独立Service，RAG接口和Agent工具共用同一检索逻辑）
 *
 * 双通道检索：
 *  1) 关键词通道：从问题提取型号token（如PR360/SA-3P），在知识库精确匹配（型号是精确标识，
 *     向量相似度会被同参数的其他型号挤掉，关键词命中保证不丢）
 *  1b) 系列通道："PR系列有哪些型号"这类问题按系列前缀跨文件取样
 *  2) 向量通道：pgvector 余弦 top8 补充语义相关段落
 */
@Service
public class ProductSearchService {

    private final VectorStore vectorStore;

    /** 型号token：字母2+，可选空格/连字符，后跟数字开头的串（PR360 / PR-341 / SA 3P / HCDP-2P / SA-3P） */
    private static final Pattern MODEL_PATTERN = Pattern.compile("[A-Z]{2,}[\\s-]?\\d[A-Z0-9]*");
    /** 系列前缀：中文"PR系列" / 英文"PR series" */
    private static final Pattern SERIES_PATTERN = Pattern.compile("([A-Z]{2,6})\\s*(?:系列|series|relays?|series relays?)", Pattern.CASE_INSENSITIVE);

    public ProductSearchService(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    /** 型号归一化：去掉空格与连字符，"SA-3P"与"SA 3P"与"SA3P"视为同一型号 */
    public static String normalizeModel(String s) {
        return s.toUpperCase().replaceAll("[\\s-]", "");
    }

    /** 主检索入口：返回最多4段去重资料 */
    public List<Document> retrieve(String msg) {
        List<Document> merged = new ArrayList<>();
        Set<String> seen = new HashSet<>();

        // 通道1：关键词——从问题提取型号token，在知识库文本中精确匹配（归一化：SA-3P ≡ SA 3P）
        List<String> modelTokens = new ArrayList<>();
        Matcher m = MODEL_PATTERN.matcher(msg);
        while (m.find()) {
            modelTokens.add(normalizeModel(m.group()));
        }
        if (!modelTokens.isEmpty()) {
            List<Document> kw = RAGConfig.KB_DOCUMENTS.stream()
                    .filter(d -> modelTokens.stream().anyMatch(t -> normalizeModel(d.getText()).contains(t)))
                    .collect(Collectors.toList());
            // 型号命中数多的排前面（如同时问多个型号）
            kw.sort((a, b) -> countHits(b.getText(), modelTokens) - countHits(a.getText(), modelTokens));
            for (Document d : kw) {
                if (merged.size() >= 4) break;
                if (seen.add(d.getText())) merged.add(d);
            }
        }

        // 通道1b：系列前缀——"PR系列有哪些型号"这类问题没有具体型号token，按前缀匹配并跨文件多样性取样
        Matcher sm = SERIES_PATTERN.matcher(msg);
        if (merged.isEmpty() && sm.find()) {
            String prefix = sm.group(1).toUpperCase();
            List<Document> series = RAGConfig.KB_DOCUMENTS.stream()
                    .filter(d -> Pattern.compile("\\b" + prefix + "-?\\d").matcher(d.getText().toUpperCase()).find())
                    .collect(Collectors.toList());
            // 按来源分组，每组取前2块（保证覆盖PR3/PR29/PR34/PR60/PR90等不同文件，而不是全被综合目录挤占）
            java.util.Map<String, List<Document>> bySource = series.stream()
                    .collect(Collectors.groupingBy(d -> String.valueOf(d.getMetadata().getOrDefault("source", "?"))));
            List<Document> sampled = new ArrayList<>();
            for (List<Document> group : bySource.values()) {
                group.subList(0, Math.min(2, group.size())).forEach(sampled::add);
            }
            sampled.sort((a, b) -> b.getText().length() - a.getText().length()); // 信息多的行优先
            for (Document d : sampled) {
                if (merged.size() >= 4) break;
                if (seen.add(d.getText())) merged.add(d);
            }
        }

        // 通道2：向量——语义相似度检索 top8
        List<Document> vec = vectorStore.similaritySearch(SearchRequest.builder()
                .query(msg)
                .similarityThreshold(0.2)
                .topK(8)
                .build());
        for (Document d : vec) {
            if (merged.size() >= 4) break;
            if (seen.add(d.getText())) merged.add(d);
        }
        return merged;
    }

    private int countHits(String text, List<String> tokens) {
        int n = 0;
        String norm = normalizeModel(text);
        for (String t : tokens) {
            if (norm.contains(t)) n++;
        }
        return n;
    }

    /** 把检索结果拼成上下文文本（供工具返回给模型） */
    public String buildContext(List<Document> docs) {
        if (docs.isEmpty()) return "（未检索到相关资料）";
        return docs.stream().map(d -> "【" + d.getMetadata().getOrDefault("source", "资料") + "】\n" + d.getText())
                .collect(Collectors.joining("\n\n---\n\n"));
    }
}
