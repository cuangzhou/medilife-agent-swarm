"""
Agent循环引擎
实现 LLM 驱动的 Skill 调用循环
支持短期记忆集成
支持约束验证（Harness Engineering）
"""
import time
import uuid
import json
from typing import Dict, Any, List, Optional
from loguru import logger

from .state_manager import StateManager, TaskStatus
from .llm_client import LLMResponse
from research.trace_graph import TraceGraphBuilder


def _emit_observability(payload: Dict[str, Any]) -> None:
    """向 EVENT_SINK 发送可观测性事件（默认 None，零开销；sink 异常不影响主流程）

    延迟导入 swarm.shared_context，避免 swarm/__init__ → agents → core.agent_loop 循环导入。
    """
    try:
        from swarm.shared_context import EVENT_SINK
        sink = EVENT_SINK.get()
        if sink is not None:
            sink(payload)
    except Exception:
        pass

# Harness Engineering: 约束验证和自动修复
try:
    from constraints import ConstraintValidator
    from validation import AutoFixer
    CONSTRAINTS_ENABLED = True
except ImportError:
    logger.warning("Constraints module not found, running without constraint validation")
    CONSTRAINTS_ENABLED = False


class AgentLoop:
    """
    Agent循环引擎
    LLM 自主决策 Skill 调用，循环直到任务完成

    功能：
    - 支持短期记忆（ShortTermMemory）
    - 自动记录每轮的 user/assistant 消息
    """

    def __init__(self, max_iterations: int = 10, short_term_memory: Optional[Any] = None, max_tool_calls: int = 2):
        """
        初始化Agent循环引擎

        Args:
            max_iterations: 最大迭代次数（防止无限循环）
            short_term_memory: 短期记忆管理器（可选）
            max_tool_calls: 最大 Skill 调用次数（硬性限制，默认2次）
        """
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.state_manager = StateManager()
        self.short_term_memory = short_term_memory
        self.tool_call_count = 0

        # Harness Engineering: 约束验证器和自动修复器
        self.validator = ConstraintValidator() if CONSTRAINTS_ENABLED else None
        self.auto_fixer = AutoFixer() if CONSTRAINTS_ENABLED else None
        if CONSTRAINTS_ENABLED:
            logger.debug("✅ Constraint validation enabled")

    async def run(self, agent, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        执行Agent循环

        Args:
            agent: Agent实例
            input_data: 输入数据

        Returns:
            最终结果
        """
        task_id = str(uuid.uuid4())
        state = self.state_manager.create_state(
            task_id=task_id,
            agent_id=agent.agent_id,
            input_data=input_data,
            max_iterations=self.max_iterations
        )

        # 重置计数
        self.tool_call_count = 0
        tool_calls_history = []
        trace_builder = TraceGraphBuilder()
        question_text = input_data.get("question") or input_data.get("query") or str(input_data)
        question_node = trace_builder.add_node(
            "question",
            "用户医学问题",
            question_text,
            source="user",
            confidence=1.0
        )
        plan_node = trace_builder.add_node(
            "plan",
            f"{agent.agent_id} 自主工具规划",
            "Agent Loop 根据上下文决定是否调用医学 Skills。",
            source=agent.agent_id,
            confidence=0.7,
            metadata={"max_tool_calls": self.max_tool_calls}
        )
        trace_builder.add_edge(question_node, plan_node, "decomposes_to", "进入 Agent Loop")

        logger.info(f"Starting Agent Loop for {agent.agent_id}, task_id={task_id}")

        try:
            state.status = TaskStatus.IN_PROGRESS

            # 初始化消息历史（包含历史对话）
            messages = self._initialize_messages(agent, input_data, session_id)

            # 记录用户消息到短期记忆
            if self.short_term_memory and session_id:
                user_message = messages[-1]["content"] if messages else str(input_data)
                self.short_term_memory.add_message(
                    session_id=session_id,
                    role="user",
                    content=user_message
                )
                logger.debug(f"Recorded user message to short-term memory (session={session_id})")

            # 获取 Agent 的 Skills (OpenAI format)
            tools_openai_format = agent.get_tools_for_llm()

            logger.debug(f"Agent has {len(tools_openai_format) if tools_openai_format else 0} skills available")

            # 主循环：LLM → Skill Calls → Results → LLM
            while state.should_continue():
                state.iteration += 1
                logger.debug(f"=== Iteration {state.iteration}/{state.max_iterations} ===")

                try:
                    # 调用 LLM（可能返回 tool_calls）
                    llm_response: LLMResponse = await agent.llm_client.chat_with_tools(
                        messages=messages,
                        tools=tools_openai_format,
                        tool_choice="auto",
                        temperature=agent.config.get('temperature', 0.7)
                    )

                    # 记录中间结果
                    state.add_intermediate_result({
                        'iteration': state.iteration,
                        'llm_response': {
                            'content': llm_response.content,
                            'tool_calls': [
                                {'name': tc.name, 'arguments': tc.arguments}
                                for tc in llm_response.tool_calls
                            ],
                            'finish_reason': llm_response.finish_reason
                        }
                    })

                    # 情况1: LLM 返回 tool_calls，执行 Skills
                    if llm_response.has_tool_calls():
                        # 硬性限制：检查是否已达到最大调用次数
                        if self.tool_call_count >= self.max_tool_calls:
                            logger.warning(f"⚠️ 已达到最大 Skill 调用次数限制 ({self.max_tool_calls})，强制生成最终答案")
                            # 强制要求 LLM 提供最终答案
                            messages.append({
                                'role': 'user',
                                'content': f'已完成 {self.max_tool_calls} 次信息检索。请基于已获取的信息提供最终答复。'
                            })
                            continue

                        logger.info(f"LLM requested {len(llm_response.tool_calls)} tool calls (当前已调用 {self.tool_call_count}/{self.max_tool_calls})")

                        # 添加 assistant 消息（包含 tool_calls）
                        messages.append(self._create_assistant_message_with_tools(llm_response))

                        # 记录 assistant 消息到短期记忆
                        if self.short_term_memory and session_id:
                            tool_names = [tc.name for tc in llm_response.tool_calls]
                            self.short_term_memory.add_message(
                                session_id=session_id,
                                role="assistant",
                                content=f"调用工具：{', '.join(tool_names)}"
                            )

                        # 执行每个 Skill 调用
                        for tool_call in llm_response.tool_calls:
                            # 增加计数
                            self.tool_call_count += 1
                            logger.debug(f"Executing: {tool_call.name}({tool_call.arguments}) - 第 {self.tool_call_count} 次调用")

                            # Harness Engineering: 验证调用
                            if self.validator:
                                validation_result = self.validator.validate_tool_call(
                                    agent.agent_id,
                                    tool_call.name
                                )
                                if not validation_result.get("valid"):
                                    logger.warning(
                                        f"⚠️ 约束警告: {validation_result.get('reason')}"
                                    )

                            _emit_observability({
                                "type": "agent_tool_call",
                                "source_agent": agent.agent_id,
                                "data": {
                                    "tool_name": tool_call.name,
                                    "arguments": str(tool_call.arguments)[:200],
                                    "call_index": self.tool_call_count,
                                },
                            })
                            _tool_started = time.monotonic()

                            tool_result = await agent.execute_tool(
                                tool_name=tool_call.name,
                                arguments=tool_call.arguments
                            )

                            result_summary = str(tool_result.get("answer", tool_result))[:500] if isinstance(tool_result, dict) else str(tool_result)[:500]
                            _emit_observability({
                                "type": "agent_tool_result",
                                "source_agent": agent.agent_id,
                                "data": {
                                    "tool_name": tool_call.name,
                                    "duration_ms": int((time.monotonic() - _tool_started) * 1000),
                                    "result_summary": result_summary[:200],
                                },
                            })
                            tool_calls_history.append({
                                "iteration": state.iteration,
                                "tool_name": tool_call.name,
                                "arguments": tool_call.arguments,
                                "result_summary": result_summary
                            })

                            retrieval_node = trace_builder.add_node(
                                "retrieval",
                                f"调用 {tool_call.name}",
                                json.dumps(tool_call.arguments, ensure_ascii=False),
                                source=agent.agent_id,
                                confidence=0.7,
                                metadata={"iteration": state.iteration, "tool_name": tool_call.name}
                            )
                            trace_builder.add_edge(plan_node, retrieval_node, "searches", "执行医学 Skill")

                            if isinstance(tool_result, dict) and tool_result.get("trace_graph"):
                                trace_builder.merge(
                                    tool_result.get("trace_graph"),
                                    prefix=f"{tool_call.name}_{self.tool_call_count}"
                                )
                            else:
                                evidence_node = trace_builder.add_node(
                                    "evidence",
                                    f"{tool_call.name} 返回结果",
                                    result_summary,
                                    source=tool_call.name,
                                    confidence=None,
                                    metadata={"iteration": state.iteration}
                                )
                                trace_builder.add_edge(retrieval_node, evidence_node, "returns", "返回工具结果")

                            # 添加结果消息
                            messages.append(
                                agent.llm_client.create_tool_message(
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_call.name,
                                    result=tool_result
                                )
                            )

                            # 记录结果到短期记忆
                            if self.short_term_memory and session_id:
                                result_summary = str(tool_result)[:200]
                                self.short_term_memory.add_message(
                                    session_id=session_id,
                                    role="tool",
                                    content=f"{tool_call.name}: {result_summary}"
                                )

                        # 继续下一轮循环
                        continue

                    # 情况2: LLM 返回文本响应，任务完成
                    else:
                        logger.info(f"LLM provided final response (no tool calls)")

                        # Harness Engineering: 验证和修复输出
                        final_answer = llm_response.content

                        if self.validator and final_answer:
                            validation_result = self.validator.validate_output(
                                agent.agent_id,
                                final_answer
                            )

                            if not validation_result.get("valid"):
                                logger.warning(
                                    f"⚠️ 输出约束违规: {validation_result.get('violations')}"
                                )

                                # 自动修复
                                if self.auto_fixer and validation_result.get("auto_fixable"):
                                    fixed_answer = self.auto_fixer.fix_output(
                                        final_answer,
                                        validation_result.get("auto_fixable", [])
                                    )
                                    if fixed_answer != final_answer:
                                        logger.info("🔧 输出已自动修复")
                                        final_answer = fixed_answer

                        # 记录最终回答到短期记忆
                        if self.short_term_memory and session_id:
                            self.short_term_memory.add_message(
                                session_id=session_id,
                                role="assistant",
                                content=final_answer or "(empty response)"
                            )
                            logger.debug(f"Recorded final answer to short-term memory (session={session_id})")

                        result = {
                            'answer': final_answer,
                            'iterations': state.iteration,
                            'agent_id': agent.agent_id,
                            'tool_calls_history': tool_calls_history
                        }

                        final_node = trace_builder.add_node(
                            "final_answer",
                            f"{agent.agent_id} 最终回答",
                            final_answer or "",
                            source=agent.agent_id,
                            confidence=0.7,
                            metadata={"iterations": state.iteration}
                        )
                        for node in list(trace_builder.nodes):
                            if node.get("type") in {"evidence", "retrieval"}:
                                trace_builder.add_edge(node["id"], final_node, "supports", "支持最终回答")
                        result["trace_graph"] = trace_builder.to_dict()

                        # 让 Agent 进行结果后处理（如提取建议等）
                        if hasattr(agent, 'post_process_result'):
                            result = await agent.post_process_result(result, final_answer)

                        state.mark_completed(result)
                        break

                except Exception as e:
                    logger.error(f"Error in iteration {state.iteration}: {e}")
                    if state.iteration >= state.max_iterations:
                        state.mark_failed(str(e))
                        break
                    # 否则继续尝试

            # 如果达到最大迭代次数但没有完成
            if not state.is_completed():
                logger.warning(f"Max iterations reached without completion")

                # 强制调用 LLM 生成最终总结
                try:
                    logger.info("Forcing LLM to provide final answer")

                    # 添加强制总结的提示
                    messages.append({
                        'role': 'user',
                        'content': '请基于以上信息，提供最终的答复。'
                    })

                    # 调用 LLM（禁用 function calling）
                    final_response = await agent.llm_client.chat_with_tools(
                        messages=messages,
                        tools=None,
                        temperature=0.7
                    )

                    result = {
                        'answer': final_response.content or '抱歉，未能完成任务',
                        'iterations': state.iteration,
                        'warning': 'max_iterations_reached',
                        'tool_calls_history': tool_calls_history
                    }
                    final_node = trace_builder.add_node(
                        "final_answer",
                        f"{agent.agent_id} 强制总结",
                        result["answer"],
                        source=agent.agent_id,
                        confidence=0.4,
                        metadata={"warning": "max_iterations_reached"}
                    )
                    trace_builder.add_edge(plan_node, final_node, "synthesizes_into", "达到最大迭代后总结")
                    result["trace_graph"] = trace_builder.to_dict()

                    # 记录最终回答到短期记忆
                    if self.short_term_memory and session_id:
                        self.short_term_memory.add_message(
                            session_id=session_id,
                            role="assistant",
                            content=result['answer']
                        )

                    state.mark_completed(result)
                    logger.info("Generated fallback answer after max iterations")

                except Exception as e:
                    logger.error(f"Failed to generate fallback answer: {e}")
                    # 降级到简单提取
                    result = {
                        'answer': '抱歉，系统在处理您的问题时遇到了问题。建议您简化问题或稍后重试。',
                        'iterations': state.iteration,
                        'warning': 'max_iterations_reached',
                        'error': str(e),
                        'tool_calls_history': tool_calls_history
                    }
                    final_node = trace_builder.add_node(
                        "final_answer",
                        f"{agent.agent_id} 降级回答",
                        result["answer"],
                        source=agent.agent_id,
                        confidence=0.2,
                        metadata={"error": str(e)}
                    )
                    trace_builder.add_edge(plan_node, final_node, "synthesizes_into", "错误降级")
                    result["trace_graph"] = trace_builder.to_dict()
                    state.mark_completed(result)

            logger.info(f"Agent Loop finished: status={state.status.value}, iterations={state.iteration}")
            return state.final_result or {}

        except Exception as e:
            logger.error(f"Agent Loop failed: {e}")
            state.mark_failed(str(e))
            raise

    def _initialize_messages(self, agent, input_data: Dict[str, Any], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """初始化消息列表，包含历史对话上下文"""
        messages = []

        # 系统提示词
        system_prompt = agent.get_system_prompt()
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        # 加载历史对话（短期记忆）
        if self.short_term_memory and session_id:
            history = self.short_term_memory.get_history(session_id, limit=5)  # 最近5轮对话
            if history:
                logger.info(f"Loaded {len(history)} historical messages from short-term memory")
                messages.extend(history)

        # 用户输入
        user_message = agent.format_user_input(input_data)
        messages.append({
            'role': 'user',
            'content': user_message
        })

        return messages

    def _create_assistant_message_with_tools(self, llm_response: LLMResponse) -> Dict[str, Any]:
        """创建包含 tool_calls 的 assistant 消息"""
        message = {
            'role': 'assistant',
            'content': llm_response.content or None
        }

        # 添加 tool_calls（OpenAI 格式）
        if llm_response.tool_calls:
            message['tool_calls'] = [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {
                        'name': tc.name,
                        'arguments': json.dumps(tc.arguments, ensure_ascii=False)
                    }
                }
                for tc in llm_response.tool_calls
            ]

        return message
