package com.example.aichatbot;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.zip.ZipFile;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformers.TransformersEmbeddingModel;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.FileSystemResource;
import org.springframework.jdbc.core.JdbcTemplate;

/**
 * RAG知识库配置：
 * ① 向量库用 PostgreSQL + pgvector 扩展（表 vector_store，重启不丢，数据在库里）
 * ② 支持喂文件：docs/ 目录下放 PDF/Word/TXT/MD，重启自动入库
 * ③ 每次启动自动加载 products.txt 产品资料
 * ④ data/vector-store.json 退化为"关键词文本缓存"（只存 text+metadata，不含 embedding，供检索侧关键词精确匹配）
 */
@Configuration
public class RAGConfig {

    private static final File STORE_FILE = new File("data/vector-store.json");
    private static final File MANIFEST_FILE = new File("data/kb-manifest.json");
    private static final File DOCS_DIR = new File("docs");
    private static final File PYTHON_SCRIPT = new File("scripts/extract_pdf_pages.py");
    /** 型号行判定：行首是字母词且（首词本身含数字 或 第二词是数字/数字+字母），兼容 PR360 / PR-341 / SA 3P / HCDP-2P */
    private static boolean isModelLine(String line) {
        String[] words = line.trim().split("\\s+");
        if (words.length == 0) return false;
        String first = words[0];
        if (!first.matches("^[A-Za-z].*")) return false;                       // 首词须字母开头
        if (first.matches("^[A-Za-z]{1,5}[-A-Za-z0-9]*\\d[-A-Za-z0-9]*$") && first.matches(".*\\d.*")) {
            return true;                                                       // PR360 / PR-341 / HCDP-2P / PDSB-60AN
        }
        return words.length > 1 && words[1].matches("^\\d+[A-Za-z]?$");        // SA 3P（空格分隔）
    }
    /** 知识库切块结果（供检索侧做关键词精确匹配，绕过向量相似度竞争） */
    public static volatile List<Document> KB_DOCUMENTS = List.of();
    private final ObjectMapper objectMapper = new ObjectMapper();

    // ① 本地向量化模型Bean（ONNX离线模型，零API成本，不依赖云端开通）
    // 多语言模型 paraphrase-multilingual-MiniLM-L12-v2：中英文通吃（中问中答/英问英答均可）
    // @Primary：排除 OpenAi 自动配置的云端 embedding（embedding.enabled=false 已禁用但 Bean 仍会注册，pgvector 需要唯一 EmbeddingModel）
    @Bean
    @Primary
    public EmbeddingModel embeddingModel() {
        TransformersEmbeddingModel model = new TransformersEmbeddingModel();
        model.setModelResource(new FileSystemResource("models/paraphrase-multilingual-MiniLM-L12-v2/model.onnx"));
        model.setTokenizerResource(new FileSystemResource("models/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json"));
        return model;
    }

    // ② 向量库Bean：由 spring-ai-starter-vector-store-pgvector 自动配置（PostgreSQL + pgvector 扩展），
    //    表 vector_store、384 维、余弦距离，见 application.yml spring.ai.vectorstore.pgvector.*
    //    这里不再自定义 Bean，直接用自动配置的 VectorStore

    // ③ 知识库加载：缓存有效（docs文件没变化）→关键词缓存直接读；docs有新增/修改→清空向量表并重新构建入库
    @Bean
    public ApplicationRunner loadKnowledgeBase(VectorStore vectorStore, JdbcTemplate jdbcTemplate) {
        return args -> {
            String manifest = buildManifest();          // 当前所有资料的指纹（文件名+大小+修改时间）
            boolean cacheValid = STORE_FILE.exists() && MANIFEST_FILE.exists()
                    && manifest.equals(Files.readString(MANIFEST_FILE.toPath()));
            if (cacheValid) {
                KB_DOCUMENTS = readDocumentsFromStore(STORE_FILE);   // 关键词匹配需要文本，从缓存文件加载
                System.out.println("[知识库] 资料无变化，向量数据在 PostgreSQL（" + countVectorRows(jdbcTemplate) + " 段），关键词缓存已加载（秒启）");
                return;
            }
            if (STORE_FILE.exists()) {
                System.out.println("[知识库] 检测到 docs/ 有新文件或修改，重新构建知识库...");
            }
            List<Document> allDocs = new ArrayList<>();
            addProductsFile(allDocs);        // 1) 内置产品资料
            loadDocsDir(allDocs);            // 2) docs/ 文件夹（PDF/Word/TXT/MD）
            if (allDocs.isEmpty()) {
                System.out.println("[知识库] 没有可加载的资料（products.txt为空且docs/文件夹不存在）");
                return;
            }
            // 3) 切块：PDF页按行切（表格每行一个型号=完整语义单元，不串行）；其他文档按token切（<512窗口）
            List<Document> chunks = splitAllDocs(allDocs);
            KB_DOCUMENTS = chunks;                                            // 暴露给检索侧做关键词匹配
            jdbcTemplate.update("DELETE FROM vector_store");                  // 清空向量表（避免旧数据残留/重复入库）
            vectorStore.add(chunks);                                          // 4) 向量化入库 PostgreSQL
            File parent = STORE_FILE.getParentFile();
            if (parent != null) parent.mkdirs();
            saveTextOnly(STORE_FILE, chunks);                                 // 5) 关键词文本缓存（不含 embedding，秒读）
            Files.writeString(MANIFEST_FILE.toPath(), manifest);              // 6) 记录文件指纹
            System.out.println("[知识库] 构建完成：" + chunks.size() + " 段资料，向量已入库 PostgreSQL（表 vector_store）");
        };
    }

    private long countVectorRows(JdbcTemplate jdbcTemplate) {
        try {
            Long n = jdbcTemplate.queryForObject("SELECT count(*) FROM vector_store", Long.class);
            return n == null ? 0 : n;
        } catch (Exception e) {
            return 0;
        }
    }

    // ---- 只存文本+metadata 的关键词缓存（不存 embedding，文件更小、读得更快）----
    private void saveTextOnly(File storeFile, List<Document> chunks) throws IOException {
        Map<String, Object> root = new HashMap<>();
        int i = 0;
        for (Document d : chunks) {
            root.put("k" + (i++), Map.of(
                    "id", d.getId(),
                    "text", d.getText(),
                    "metadata", d.getMetadata() == null ? Map.of() : d.getMetadata()));
        }
        objectMapper.writeValue(storeFile, root);
    }

    // ---- 生成文件指纹清单：用于判断docs是否变化（新增/删除/修改任一文件都会触发重建）----
    private String buildManifest() throws IOException {
        StringBuilder sb = new StringBuilder();
        ClassPathResource products = new ClassPathResource("products.txt");
        if (products.exists()) {
            sb.append("products.txt|").append(products.contentLength()).append('\n');
        }
        if (DOCS_DIR.exists()) {
            try (Stream<Path> files = Files.walk(DOCS_DIR.toPath())) {
                files.filter(Files::isRegularFile).sorted().forEach(p -> {
                    try {
                        sb.append(p.getFileName()).append('|')
                          .append(Files.size(p)).append('|')
                          .append(Files.getLastModifiedTime(p).toMillis()).append('\n');
                    } catch (IOException ignored) {
                    }
                });
            }
        }
        return sb.toString();
    }

    // ---- 加载内置产品资料 ----
    private void addProductsFile(List<Document> allDocs) throws IOException {
        ClassPathResource resource = new ClassPathResource("products.txt");
        if (resource.exists()) {
            TextReader reader = new TextReader(resource);
            reader.setCharset(StandardCharsets.UTF_8);
            allDocs.addAll(reader.get());
        }
    }

    // ---- 扫描 docs/ 目录，按扩展名解析 ----
    private void loadDocsDir(List<Document> allDocs) throws IOException {
        if (!DOCS_DIR.exists()) {
            Files.createDirectories(DOCS_DIR.toPath());
            System.out.println("[知识库] 已创建 docs/ 文件夹：把 PDF/Word/TXT/MD 放进去，重启即自动入库");
            return;
        }
        try (Stream<Path> files = Files.walk(DOCS_DIR.toPath())) {
            files.filter(Files::isRegularFile).forEach(p -> {
                File f = p.toFile();
                String name = f.getName().toLowerCase();
                try {
                    if (name.endsWith(".txt") || name.endsWith(".md")) {
                        TextReader reader = new TextReader(new FileSystemResource(f));
                        reader.setCharset(StandardCharsets.UTF_8);
                        allDocs.addAll(reader.get());
                        System.out.println("[知识库] 读取文本: " + name);
                    } else if (name.endsWith(".pdf")) {
                        // 按页切块：一页一个完整语义单元（型号表+参数不分离，表格型PDF检索更准）
                        allDocs.addAll(extractPdfPages(f));
                        System.out.println("[知识库] 读取PDF: " + name);
                    } else if (name.endsWith(".docx")) {
                        allDocs.add(new Document(extractDocxText(f), Map.of("source", name)));
                        System.out.println("[知识库] 读取Word: " + name);
                    }
                } catch (Exception e) {
                    System.out.println("[知识库] 跳过文件 " + name + "（解析失败: " + e.getMessage() + "）");
                }
            });
        }
    }

    // ---- 切块入口：PDF页按行切，其他文档按token切 ----
    private List<Document> splitAllDocs(List<Document> allDocs) {
        List<Document> chunks = new ArrayList<>();
        for (Document doc : allDocs) {
            if ("pdf".equals(doc.getMetadata().get("format"))) {
                chunks.addAll(splitPdfPageByLines(doc));
            } else {
                chunks.addAll(new TokenTextSplitter(400, 200, 50, 10000, true,
                        List.of('\n', ' ', '.', ',', ';', '!', '?')).split(List.of(doc)));
            }
        }
        return chunks;
    }

    // ---- PDF页按行切块：型号行（含字母+数字）单独成块=完整语义单元（不加表头，避免均值池化稀释型号信号）；段落行累积，每8行或350字符一块 ----
    private List<Document> splitPdfPageByLines(Document doc) {
        List<Document> chunks = new ArrayList<>();
        StringBuilder paragraph = new StringBuilder();
        int paragraphLines = 0;
        for (String rawLine : doc.getText().split("\\r?\\n")) {
            String line = rawLine.trim();
            if (line.isEmpty()) continue;
            if (isModelLine(line)) {
                flushParagraph(paragraph, doc, chunks);
                chunks.add(new Document(line, doc.getMetadata()));
            } else {
                paragraph.append(line).append('\n');
                if (++paragraphLines >= 8 || paragraph.length() > 350) {
                    flushParagraph(paragraph, doc, chunks);
                }
            }
        }
        flushParagraph(paragraph, doc, chunks);
        return chunks;
    }

    private void flushParagraph(StringBuilder paragraph, Document doc, List<Document> chunks) {
        if (paragraph.length() > 0) {
            chunks.add(new Document(paragraph.toString(), doc.getMetadata()));
            paragraph.setLength(0);
        }
    }

    // ---- 从向量库JSON读取文档（缓存加载时恢复关键词匹配文本）----
    private List<Document> readDocumentsFromStore(File storeFile) {
        try {
            JsonNode root = objectMapper.readTree(storeFile);
            List<Document> docs = new ArrayList<>();
            root.fields().forEachRemaining(e -> {
                JsonNode node = e.getValue();
                Map<String, Object> meta = new HashMap<>();
                node.path("metadata").fields().forEachRemaining(m ->
                        meta.put(m.getKey(), m.getValue().asText()));
                docs.add(new Document(node.path("text").asText(), meta));
            });
            return docs;
        } catch (IOException ex) {
            System.out.println("[知识库] 读取缓存文档失败：" + ex.getMessage());
            return List.of();
        }
    }

    // ---- PDF文本提取：优先 PyMuPDF（CAD/排版软件生成的PDF不丢数字），失败回退 PDFBox ----
    private List<Document> extractPdfPages(File file) throws IOException {
        try {
            return extractPdfPagesWithPyMuPDF(file);
        } catch (Exception e) {
            System.out.println("[知识库] PyMuPDF 提取失败（" + e.getMessage() + "），回退 PDFBox：" + file.getName());
            return extractPdfPagesWithPdfBox(file);
        }
    }

    // ---- PyMuPDF：调 python scripts/extract_pdf_pages.py <pdf>，逐页提取文本（sort=True 按阅读顺序）----
    private List<Document> extractPdfPagesWithPyMuPDF(File file) throws IOException {
        if (!PYTHON_SCRIPT.exists()) {
            throw new IOException("脚本不存在: " + PYTHON_SCRIPT.getAbsolutePath());
        }
        ProcessBuilder pb = new ProcessBuilder("python", PYTHON_SCRIPT.getAbsolutePath(), file.getAbsolutePath());
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String out;
        try (InputStream is = p.getInputStream()) {
            out = new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
        int exit;
        try {
            exit = p.waitFor();
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new IOException("PyMuPDF 脚本被中断", ie);
        }
        if (exit != 0) {
            throw new IOException("脚本退出码 " + exit + ": " + out);
        }
        JsonNode root = objectMapper.readTree(out);
        if (root.has("error")) {
            throw new IOException(root.get("error").asText());
        }
        List<Document> pages = new ArrayList<>();
        for (JsonNode node : root.path("pages")) {
            String text = node.path("text").asText();
            if (text != null && !text.isBlank()) {
                pages.add(new Document(text, Map.of("source", file.getName(), "page", node.path("page").asInt(), "format", "pdf")));
            }
        }
        if (pages.isEmpty()) {
            throw new IOException("脚本未提取到任何文本");
        }
        return pages;
    }

    // ---- PDFBox 兜底：按页提取，每页一个Document（型号表+参数保持同一页，语义完整）----
    private List<Document> extractPdfPagesWithPdfBox(File file) throws IOException {
        List<Document> pages = new ArrayList<>();
        try (PDDocument doc = Loader.loadPDF(file)) {
            PDFTextStripper stripper = new PDFTextStripper();
            for (int i = 1; i <= doc.getNumberOfPages(); i++) {
                stripper.setStartPage(i);
                stripper.setEndPage(i);
                String text = stripper.getText(doc);
                if (text != null && !text.isBlank()) {
                    pages.add(new Document(text, Map.of("source", file.getName(), "page", i, "format", "pdf")));
                }
            }
        }
        return pages;
    }

    // ---- Word(docx)文本提取：docx本质是zip，直接读word/document.xml ----
    private String extractDocxText(File file) throws IOException {
        try (ZipFile zip = new ZipFile(file)) {
            var entry = zip.getEntry("word/document.xml");
            if (entry == null) return "";
            String xml = new String(zip.getInputStream(entry).readAllBytes(), StandardCharsets.UTF_8);
            // 去掉XML标签，保留段落换行
            return xml.replaceAll("</w:p>", "\n").replaceAll("<[^>]+>", "");
        }
    }
}
