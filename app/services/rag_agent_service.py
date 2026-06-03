"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_qwq import ChatQwen

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


SUMMARY_THRESHOLD = 10
SUMMARY_KEEP_RECENT = 6
SUMMARY_PROMPT = """请将以下对话历史浓缩成一段简洁的摘要，保留关键信息、用户意图和重要结论。

对话历史：
{conversation}

要求：
1. 保留用户的核心问题和需求
2. 保留重要的结论和决策
3. 保留关键的事实信息
4. 使用简洁的中文表达
5. 摘要长度控制在200字以内

摘要："""


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 创建内存检查点（用于会话管理）
        self.checkpointer = MemorySaver()

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _summarize_messages(
        self,
        messages: Sequence[BaseMessage]
    ) -> str:
        """
        将早期对话消息浓缩成摘要

        Args:
            messages: 需要摘要的消息列表

        Returns:
            str: 对话摘要
        """
        conversation_text = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            conversation_text.append(f"{role}: {content}")

        conversation = "\n".join(conversation_text)

        prompt = SUMMARY_PROMPT.format(conversation=conversation)

        try:
            summary_model = ChatQwen(
                model=self.model_name,
                api_key=config.dashscope_api_key,
                temperature=0.3,
                streaming=False,
            )

            response = await summary_model.ainvoke([HumanMessage(content=prompt)])
            summary = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"对话摘要生成成功，原始消息: {len(messages)} 条，摘要长度: {len(summary)} 字")
            return summary

        except Exception as e:
            logger.error(f"生成对话摘要失败: {e}")
            return f"[早期对话摘要生成失败，共{len(messages)}条消息]"

    async def _get_and_compress_history(
        self,
        session_id: str
    ) -> list[BaseMessage]:
        """
        获取会话历史并在需要时进行压缩（摘要浓缩）

        策略：
        - 消息数 <= SUMMARY_THRESHOLD: 返回全部历史
        - 消息数 > SUMMARY_THRESHOLD: 将早期对话浓缩成摘要，保留最近的对话

        Args:
            session_id: 会话ID

        Returns:
            list[BaseMessage]: 处理后的消息列表
        """
        try:
            checkpointer_config = {"configurable": {"thread_id": session_id}}
            checkpoint_tuple = self.checkpointer.get(checkpointer_config)

            logger.debug(f"[会话 {session_id}] checkpoint_tuple type: {type(checkpoint_tuple)}, value: {checkpoint_tuple}")

            if not checkpoint_tuple:
                logger.debug(f"[会话 {session_id}] checkpoint_tuple 为空，返回空历史")
                return []

            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint
                logger.debug(f"[会话 {session_id}] 使用 .checkpoint 属性")
            elif isinstance(checkpoint_tuple, tuple) and len(checkpoint_tuple) > 0:
                checkpoint_data = checkpoint_tuple[0]
                if not isinstance(checkpoint_data, dict):
                    checkpoint_data = {}
                logger.debug(f"[会话 {session_id}] 使用 tuple[0]")
            else:
                checkpoint_data = {}
                logger.debug(f"[会话 {session_id}] 使用空 dict")

            if not isinstance(checkpoint_data, dict):
                checkpoint_data = {}

            channel_values = checkpoint_data.get("channel_values", {})
            if not isinstance(channel_values, dict):
                channel_values = {}
            messages = channel_values.get("messages", [])

            logger.info(f"[会话 {session_id}] checkpoint_data keys: {list(checkpoint_data.keys()) if isinstance(checkpoint_data, dict) else 'N/A'}")
            logger.info(f"[会话 {session_id}] channel_values keys: {list(channel_values.keys()) if isinstance(channel_values, dict) else 'N/A'}")
            logger.info(f"[会话 {session_id}] messages count: {len(messages)}")

            if not messages:
                logger.info(f"[会话 {session_id}] messages 为空，返回空历史")
                return []

            if len(messages) <= SUMMARY_THRESHOLD:
                logger.debug(f"[会话 {session_id}] 历史消息数: {len(messages)}，无需压缩")
                return list(messages)

            system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None

            start_idx = 1 if system_msg else 0
            old_messages = messages[start_idx:-SUMMARY_KEEP_RECENT]
            recent_messages = messages[-SUMMARY_KEEP_RECENT:]

            logger.info(f"[会话 {session_id}] 开始压缩历史，总消息: {len(messages)}，早期: {len(old_messages)}，保留: {len(recent_messages)}")

            summary = await self._summarize_messages(old_messages)

            summary_message = SystemMessage(
                content=f"【早期对话摘要】\n{summary}"
            )

            new_messages = []
            if system_msg:
                new_messages.append(system_msg)
            new_messages.append(summary_message)
            new_messages.extend(recent_messages)

            self._update_checkpointer(session_id, new_messages)

            logger.info(f"[会话 {session_id}] 历史压缩完成，{len(messages)} -> {len(new_messages)} 条消息")

            return new_messages

        except Exception as e:
            logger.error(f"[会话 {session_id}] 获取/压缩历史失败: {e}")
            return []

    def _update_checkpointer(
        self,
        session_id: str,
        messages: list[BaseMessage]
    ):
        """
        更新 checkpointer 中的消息历史

        Args:
            session_id: 会话ID
            messages: 新的消息列表
        """
        try:
            config = {"configurable": {"thread_id": session_id}}

            checkpoint_tuple = self.checkpointer.get(config)
            if not checkpoint_tuple:
                return

            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = dict(checkpoint_tuple.checkpoint)
            else:
                checkpoint_data = dict(checkpoint_tuple[0]) if checkpoint_tuple else {}

            channel_values = checkpoint_data.get("channel_values", {})
            channel_values["messages"] = messages
            checkpoint_data["channel_values"] = channel_values

            if hasattr(checkpoint_tuple, 'metadata'):
                metadata = checkpoint_tuple.metadata
            else:
                metadata = checkpoint_tuple[1] if len(checkpoint_tuple) > 1 else {}

            self.checkpointer.put(config, checkpoint_data, metadata)

        except Exception as e:
            logger.error(f"[会话 {session_id}] 更新 checkpointer 失败: {e}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        all_tools = list(self.tools)
        
        try:
            mcp_client = await get_mcp_client_with_retry()
            mcp_tools = await mcp_client.get_tools()
            logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")
            self.mcp_tools = mcp_tools
            all_tools.extend(mcp_tools)
        except Exception as e:
            logger.warning(f"MCP 工具加载失败: {e}，将仅使用基础工具")
            self.mcp_tools = []

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. **重要**: 当用户提出任何问题时，首先调用 retrieve_knowledge 工具从知识库中检索相关信息
            2. 理解用户需求，选择合适的工具来完成任务
            3. 当需要获取实时信息或专业知识时，主动使用相关工具
            4. 基于工具返回的结果提供准确、专业的回答
            5. 如果知识库中没有相关信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明
            - 优先使用知识库中的信息回答问题

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            await self._get_and_compress_history(session_id)

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            agent_input = {"messages": messages}

            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（非流式）: {e}")
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            await self._get_and_compress_history(session_id)

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            agent_input = {"messages": messages}

            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
            raise

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 MemorySaver checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            # 使用 checkpointer 的 get 方法获取最新的检查点
            config = {"configurable": {"thread_id": session_id}}
            
            # 获取该 thread 的最新检查点
            checkpoint_tuple = self.checkpointer.get(config)
            
            if not checkpoint_tuple:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []
            
            # checkpoint_tuple 可能是命名元组或普通元组，安全地提取 checkpoint
            # 通常第一个元素是 checkpoint 数据
            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint  # type: ignore
            else:
                # 如果是普通元组，第一个元素是 checkpoint
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}
            
            # 从检查点中提取消息
            messages = checkpoint_data.get("channel_values", {}).get("messages", [])
            
            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue
                    
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                
                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })
            
            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history
            
        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 MemorySaver checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            # 使用 checkpointer 的 delete_thread 方法删除该 thread 的所有检查点
            self.checkpointer.delete_thread(session_id)
            
            logger.info(f"已清除会话历史: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)
