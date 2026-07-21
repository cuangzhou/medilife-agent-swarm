"""
Skill 注册系统
直接将 Skill 函数转换为 OpenAI function calling 格式
"""
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
import inspect
import asyncio
from loguru import logger
from pydantic import BaseModel, ConfigDict, RootModel, StrictBool, StrictFloat, StrictInt, StrictStr, ValidationError, create_model


@dataclass
class SkillParameter:
    """Skill 参数定义"""
    name: str
    type: str  # "string", "number", "integer", "boolean", "object", "array"
    description: str
    required: bool = False
    enum: Optional[List[str]] = None


class SkillOutput(RootModel[Dict[str, Any]]):
    """Common output contract for all registered Skills."""


_PYDANTIC_TYPES = {
    "string": StrictStr,
    "number": StrictFloat,
    "integer": StrictInt,
    "boolean": StrictBool,
    "object": Dict[str, Any],
    "array": List[Any],
}


def _build_input_model(name: str, parameters: List[SkillParameter]) -> type[BaseModel]:
    fields: Dict[str, Any] = {}
    for parameter in parameters:
        annotation = _PYDANTIC_TYPES.get(parameter.type, Any)
        default = ... if parameter.required else None
        if not parameter.required:
            annotation = Optional[annotation]
        fields[parameter.name] = (annotation, default)
    return create_model(
        f"{''.join(part.title() for part in name.split('_'))}Input",
        __config__=ConfigDict(extra="forbid", strict=True),
        **fields,
    )


class SkillRegistry:
    """
    Skill 注册表

    存储 Skill 函数并提供执行和格式转换能力
    """

    def __init__(self):
        self.skills: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: List[SkillParameter]
    ):
        """
        注册 Skill

        Args:
            name: Skill 名称
            function: Skill 函数（async 或 sync）
            description: Skill 描述
            parameters: 参数列表
        """
        self.skills[name] = {
            'function': function,
            'description': description,
            'parameters': parameters,
            'is_async': inspect.iscoroutinefunction(function),
            'input_model': _build_input_model(name, parameters),
            'output_model': SkillOutput,
        }
        logger.debug(f"Registered skill: {name}")

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """获取 Skill"""
        return self.skills.get(name)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """获取所有 Skills"""
        return self.skills

    async def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        执行 Skill

        Args:
            name: Skill 名称
            **kwargs: Skill 参数

        Returns:
            Skill 执行结果
        """
        skill = self.skills.get(name)
        if not skill:
            error_msg = f"Skill not found: {name}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

        try:
            validated_input = skill['input_model'].model_validate(kwargs).model_dump(exclude_none=True)
            logger.debug(f"Executing skill: {name} with args: {validated_input}")

            if skill['is_async']:
                # Async skill
                result = await skill['function'](**validated_input)
            else:
                # Sync skill - run in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: skill['function'](**validated_input)
                )

            result = skill['output_model'].model_validate(result).root
            logger.debug(f"Skill {name} completed successfully")
            return result

        except ValidationError as e:
            logger.warning(f"Skill validation failed: {name} - {e}")
            return {
                "success": False,
                "status": "validation_error",
                "error": "Skill input or output did not satisfy its Pydantic contract",
                "skill": name,
                "details": e.errors(include_url=False),
            }
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Optional capability unavailable for {name}: {e}")
            return {
                "success": False,
                "status": "unavailable",
                "error": str(e),
                "skill": name,
            }
        except Exception as e:
            error_msg = f"Skill execution failed: {name} - {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "skill": name
            }

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """
        直接转换为 OpenAI function calling 格式

        Returns:
            OpenAI tools 格式的列表
        """
        tools = []

        for name, skill in self.skills.items():
            schema = skill['input_model'].model_json_schema()
            properties = schema.get('properties', {})
            required = schema.get('required', [])

            for param in skill['parameters']:
                prop = properties.setdefault(param.name, {})
                prop['description'] = param.description
                if param.enum:
                    prop['enum'] = param.enum

            tools.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'description': skill['description'],
                    'parameters': {
                        'type': 'object',
                        'properties': properties,
                        'required': required,
                        'additionalProperties': False,
                    }
                }
            })

        return tools
