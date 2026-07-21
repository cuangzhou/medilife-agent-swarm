from swarm.events import Event, EventType
from swarm.shared_context import EVENT_SINK, SharedContext, SubTask
from api_server import health_service_metadata


def test_event_sink_is_request_scoped_and_description_is_present():
    received = []
    token = EVENT_SINK.set(received.append)
    try:
        context = SharedContext("session-test")
        context.add_subtask(SubTask(id="s1", type="research", description="检索过敏证据", assigned_agent="research_agent"))
    finally:
        EVENT_SINK.reset(token)
    assert received[0]["data"]["description"] == "检索过敏证据"
    assert EVENT_SINK.get() is None


def test_event_sink_failure_does_not_break_publish():
    token = EVENT_SINK.set(lambda _: (_ for _ in ()).throw(RuntimeError("observer failed")))
    try:
        context = SharedContext("session-test")
        context.publish_event(Event(type=EventType.SWARM_STARTED, source_agent="lead", data={}))
        assert len(context.events) == 1
    finally:
        EVENT_SINK.reset(token)


def test_health_exposes_only_non_sensitive_model_metadata():
    health = health_service_metadata()
    assert "model" in health and "baseUrl" in health
    assert "api_key" not in health
