package com.example.aichatbot;

import java.util.ArrayList;
import java.util.List;

import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import jakarta.annotation.PostConstruct;

/**
 * 多轮对话记忆（课3）：把聊天历史持久化到 PostgreSQL（表 chat_history）
 *
 * 实现 Spring AI 原生 ChatMemory 接口：
 *   add(conversationId, messages)  写历史
 *   get(conversationId, lastN)     读最近N条
 *   clear(conversationId)          清空
 *
 * 比 Spring AI 默认的 InMemoryChatMemory 强在：重启不丢（存数据库），
 * 面试价值：知道"内存记忆 vs 持久化记忆"的取舍，知道用数据库做上下文管理。
 */
@Component
public class PersistentChatMemory implements ChatMemory {

    private final JdbcTemplate jdbcTemplate;

    public PersistentChatMemory(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @PostConstruct
    public void initTable() {
        jdbcTemplate.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id              BIGSERIAL PRIMARY KEY,
                    conversation_id VARCHAR(64)  NOT NULL,
                    role            VARCHAR(16)  NOT NULL,
                    content         TEXT         NOT NULL,
                    created_at      TIMESTAMP    DEFAULT now()
                )
                """);
        jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_cid ON chat_history (conversation_id, id)");
    }

    @Override
    public void add(String conversationId, List<Message> messages) {
        for (Message m : messages) {
            String role = m instanceof UserMessage ? "USER" : "ASSISTANT";
            jdbcTemplate.update(
                    "INSERT INTO chat_history (conversation_id, role, content) VALUES (?, ?, ?)",
                    conversationId, role, m.getText());
        }
    }

    @Override
    public List<Message> get(String conversationId) {
        List<Message> result = new ArrayList<>();
        // 取最近20条（按 id 倒序），再反转成正序（最早在前），避免会话无限增长
        List<Message> recent = jdbcTemplate.query(
                "SELECT role, content FROM chat_history WHERE conversation_id = ? ORDER BY id DESC LIMIT 20",
                (rs, i) -> {
                    String role = rs.getString("role");
                    String content = rs.getString("content");
                    return "USER".equals(role) ? new UserMessage(content) : new AssistantMessage(content);
                },
                conversationId);
        for (int i = recent.size() - 1; i >= 0; i--) {
            result.add(recent.get(i));
        }
        return result;
    }

    @Override
    public void clear(String conversationId) {
        jdbcTemplate.update("DELETE FROM chat_history WHERE conversation_id = ?", conversationId);
    }
}
