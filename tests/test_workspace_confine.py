"""Workspace confinement: file tools are hard-bounded to the workspace folder
(layered on upstream's sensitive-path policy); bash runs with cwd there."""
import os
import tempfile

import pytest

from src.tool_execution import _resolve_tool_path_in_workspace, _direct_fallback


def test_workspace_resolver_confines():
    ws = tempfile.mkdtemp()
    open(os.path.join(ws, "a.txt"), "w").write("x")
    real = os.path.realpath(os.path.join(ws, "a.txt"))
    # relative path resolves under the workspace
    assert _resolve_tool_path_in_workspace(ws, "a.txt") == real
    # absolute path inside the workspace is allowed
    assert _resolve_tool_path_in_workspace(ws, os.path.join(ws, "a.txt")) == real
    # absolute path outside is rejected (sibling temp dir, portable across OSes)
    outside = tempfile.mkdtemp()
    with pytest.raises(ValueError):
        _resolve_tool_path_in_workspace(ws, os.path.join(outside, "x.txt"))
    # parent-escape is rejected
    with pytest.raises(ValueError):
        _resolve_tool_path_in_workspace(ws, os.path.join("..", "..", "escape.txt"))


def test_workspace_resolver_blocks_sensitive():
    """Upstream's sensitive-file deny list still applies inside the workspace."""
    ws = tempfile.mkdtemp()
    os.makedirs(os.path.join(ws, ".ssh"), exist_ok=True)
    with pytest.raises(ValueError):
        _resolve_tool_path_in_workspace(ws, ".ssh/authorized_keys")


@pytest.mark.asyncio
async def test_read_write_confined_in_workspace():
    ws = tempfile.mkdtemp()
    # Write inside the workspace (relative path) succeeds.
    res = await _direct_fallback("write_file", "note.txt\nhello", workspace=ws)
    assert res["exit_code"] == 0
    assert os.path.isfile(os.path.join(ws, "note.txt"))
    # Read it back.
    res = await _direct_fallback("read_file", "note.txt", workspace=ws)
    assert res["exit_code"] == 0 and res["output"] == "hello"
    # Reading outside the workspace is rejected (sibling temp dir, portable).
    outside = tempfile.mkdtemp()
    outside_file = os.path.join(outside, "secret.txt")
    open(outside_file, "w").write("nope")
    res = await _direct_fallback("read_file", outside_file, workspace=ws)
    assert res["exit_code"] == 1 and "outside the workspace" in res["error"]
    # Writing outside is rejected (file must not be created).
    escape = os.path.join(outside, "_ws_escape.txt")
    res = await _direct_fallback("write_file", f"{escape}\nx", workspace=ws)
    assert res["exit_code"] == 1 and "outside the workspace" in res["error"]
    assert not os.path.exists(escape)


@pytest.mark.asyncio
async def test_subprocess_runs_with_workspace_cwd():
    """bash/python subprocesses run with cwd set to the workspace. Use the
    python tool for an OS-agnostic cwd probe (Windows cmd has no `pwd`)."""
    ws = tempfile.mkdtemp()
    res = await _direct_fallback("python", "import os; print(os.getcwd())", workspace=ws)
    assert res["exit_code"] == 0
    assert os.path.realpath(res["output"].strip()) == os.path.realpath(ws)


# --- Tools that landed after this PR, now wired into the workspace -----------

@pytest.mark.asyncio
async def test_edit_file_confined_in_workspace():
    import json
    ws = tempfile.mkdtemp()
    open(os.path.join(ws, "f.txt"), "w").write("foo bar")
    # Edit inside the workspace succeeds.
    res = await _direct_fallback("edit_file", json.dumps(
        {"path": "f.txt", "old_string": "foo", "new_string": "baz"}), workspace=ws)
    assert res["exit_code"] == 0
    assert open(os.path.join(ws, "f.txt")).read() == "baz bar"
    # Editing outside the workspace is rejected (sibling temp dir, portable).
    outside = tempfile.mkdtemp()
    outside_file = os.path.join(outside, "f.txt")
    open(outside_file, "w").write("a")
    res = await _direct_fallback("edit_file", json.dumps(
        {"path": outside_file, "old_string": "a", "new_string": "b"}), workspace=ws)
    assert res["exit_code"] == 1 and "outside the workspace" in res["error"]


@pytest.mark.asyncio
async def test_grep_and_ls_confined_in_workspace():
    import json
    ws = tempfile.mkdtemp()
    open(os.path.join(ws, "doc.txt"), "w").write("hello workspace\n")
    # grep with no path searches the workspace root and finds the match.
    res = await _direct_fallback("grep", json.dumps({"pattern": "hello"}), workspace=ws)
    assert res["exit_code"] == 0 and "doc.txt" in res["output"]
    # grep pointed outside the workspace is rejected (sibling temp dir, portable).
    outside = tempfile.mkdtemp()
    res = await _direct_fallback("grep", json.dumps({"pattern": "x", "path": outside}), workspace=ws)
    assert res["exit_code"] == 1 and "outside the workspace" in res["error"]
    # ls of the workspace lists its files; ls outside is rejected.
    res = await _direct_fallback("ls", "", workspace=ws)
    assert res["exit_code"] == 0 and "doc.txt" in res["output"]
    res = await _direct_fallback("ls", outside, workspace=ws)
    assert res["exit_code"] == 1 and "outside the workspace" in res["error"]


@pytest.mark.asyncio
async def test_workspace_tool_routing_bypasses_mcp(monkeypatch):
    from src.tool_execution import execute_tool_block
    import src.tool_execution as tool_exec

    class DummyBlock:
        def __init__(self, tool_type, content):
            self.tool_type = tool_type
            self.content = content

    called_mcp = []
    called_fallback = []

    async def mock_call_mcp_tool(tool, content, progress_cb=None, workspace=None):
        called_mcp.append(tool)
        return {"output": "mcp", "exit_code": 0}

    async def mock_direct_fallback(tool, content, progress_cb=None, workspace=None):
        called_fallback.append(tool)
        return {"output": "fallback", "exit_code": 0}

    monkeypatch.setattr(tool_exec, "_call_mcp_tool", mock_call_mcp_tool)
    monkeypatch.setattr(tool_exec, "_direct_fallback", mock_direct_fallback)
    monkeypatch.setattr(tool_exec, "_owner_is_admin", lambda owner: True)

    # 1. Without workspace, read_file should go to MCP
    block = DummyBlock(tool_type="read_file", content="somefile.txt")
    called_mcp.clear()
    called_fallback.clear()
    await execute_tool_block(block, workspace=None)
    assert "read_file" in called_mcp
    assert not called_fallback

    # 2. With workspace, read_file should bypass MCP and go to direct fallback
    called_mcp.clear()
    called_fallback.clear()
    await execute_tool_block(block, workspace="/some/path")
    assert not called_mcp
    assert "read_file" in called_fallback

    # 3. With workspace, a tool like web_search should still go to MCP
    block_web = DummyBlock(tool_type="web_search", content="query")
    called_mcp.clear()
    called_fallback.clear()
    await execute_tool_block(block_web, workspace="/some/path")
    assert "web_search" in called_mcp
    assert not called_fallback


@pytest.mark.asyncio
async def test_chat_stream_workspace_disables_docs(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from routes.chat_routes import setup_chat_routes
    from unittest.mock import MagicMock, AsyncMock

    # Mock all setup_chat_routes dependencies
    session_manager = MagicMock()
    chat_handler = MagicMock()
    chat_handler.handle_memory_command = AsyncMock(return_value=None)
    chat_processor = MagicMock()
    memory_manager = MagicMock()
    research_handler = MagicMock()
    upload_handler = MagicMock()
    skills_manager = MagicMock()

    mock_session = MagicMock()
    mock_session.name = "My Session"
    mock_session.model = "gpt-4o"
    mock_session.endpoint_url = "https://api.openai.com/v1"
    mock_session.headers = {"Authorization": "Bearer key"}
    mock_session.history = []
    session_manager.get_session.return_value = mock_session

    # Mock gates/auth
    monkeypatch.setattr("routes.chat_routes.get_current_user", lambda req: "alice")
    monkeypatch.setattr("routes.chat_routes._verify_session_owner", lambda req, sess: None)
    monkeypatch.setattr("routes.chat_routes._set_user_time_from_request", lambda req: None)
    monkeypatch.setattr("routes.chat_routes._clear_orphaned_session_endpoint", lambda sess, owner: False)
    monkeypatch.setattr("routes.chat_routes._recover_empty_session_model", lambda sess, session, owner: None)
    monkeypatch.setattr("routes.chat_routes._enforce_chat_privileges", lambda req, sess: None)
    monkeypatch.setattr("routes.chat_routes.resolve_session_auth", lambda sess, session, owner: None)
    monkeypatch.setattr("routes.chat_routes.set_session_mode", lambda session, mode: None)
    monkeypatch.setattr("routes.chat_routes.get_session_mode", lambda session: None)

    # Make sure os.path.isdir returns True for our mock workspace path
    monkeypatch.setattr("os.path.isdir", lambda path: True if path == "/mock/workspace" else False)
    monkeypatch.setattr("os.path.realpath", lambda path: path)

    # Capture the build_effective_tool_policy calls and check disabled_tools
    captured_disabled_tools = []

    def mock_build_effective_tool_policy(disabled_tools=None, last_user_message=None):
        captured_disabled_tools.append(disabled_tools)
        mock_policy = MagicMock()
        mock_policy.block_all_tool_calls = False
        mock_policy.blocks.return_value = False
        mock_policy.all_disabled_names.return_value = disabled_tools or set()
        return mock_policy

    monkeypatch.setattr("routes.chat_routes.build_effective_tool_policy", mock_build_effective_tool_policy)

    async def mock_build_chat_context(*args, **kwargs):
        mock_ctx = MagicMock()
        mock_ctx.web_sources = []
        mock_ctx.rag_sources = []
        mock_ctx.auto_opened_docs = []
        mock_ctx.used_memories = []
        mock_ctx.was_compacted = False
        mock_ctx.context_length = 1000
        mock_ctx.preprocessed = MagicMock()
        mock_ctx.preprocessed.attachment_meta = None
        mock_ctx.user = "alice"
        mock_ctx.preset = MagicMock()
        mock_ctx.preset.temperature = 0.7
        mock_ctx.preset.max_tokens = 1000
        mock_ctx.preset.character_name = None
        return mock_ctx

    monkeypatch.setattr("routes.chat_routes.build_chat_context", mock_build_chat_context)

    async def mock_stream_llm(*args, **kwargs):
        yield "data: [DONE]\n\n"

    async def mock_stream_agent_loop(*args, **kwargs):
        yield "data: [DONE]\n\n"

    monkeypatch.setattr("routes.chat_routes.stream_llm_with_fallback", mock_stream_llm)
    monkeypatch.setattr("routes.chat_routes.stream_agent_loop", mock_stream_agent_loop)
    monkeypatch.setattr("routes.chat_routes.save_assistant_response", lambda *args, **kwargs: "mock-msg-id")
    monkeypatch.setattr("routes.chat_routes.run_post_response_tasks", lambda *args, **kwargs: None)

    # Setup FastAPI app
    app = FastAPI()
    app.include_router(setup_chat_routes(
        session_manager=session_manager,
        chat_handler=chat_handler,
        chat_processor=chat_processor,
        memory_manager=memory_manager,
        research_handler=research_handler,
        upload_handler=upload_handler,
        skills_manager=skills_manager
    ))
    client = TestClient(app)

    # Call chat_stream without workspace
    response = client.post(
        "/api/chat_stream",
        data={
            "message": "hello",
            "session": "session-1",
            "workspace": "",
            "allow_bash": "true",
            "allow_web_search": "true"
        }
    )
    assert response.status_code == 200
    doc_tools = {"create_document", "edit_document", "update_document", "suggest_document", "manage_documents"}
    assert captured_disabled_tools
    last_disabled = None
    for dt in captured_disabled_tools:
        if dt is not None:
            last_disabled = dt
    if last_disabled is None:
        last_disabled = set()
    assert not (doc_tools & last_disabled)

    # Call chat_stream with active workspace
    captured_disabled_tools.clear()
    response = client.post(
        "/api/chat_stream",
        data={
            "message": "hello",
            "session": "session-1",
            "workspace": "/mock/workspace",
            "allow_bash": "true",
            "allow_web_search": "true"
        }
    )
    assert response.status_code == 200
    assert captured_disabled_tools
    last_disabled = None
    for dt in captured_disabled_tools:
        if dt is not None:
            last_disabled = dt
    if last_disabled is None:
        last_disabled = set()
    assert doc_tools.issubset(last_disabled)

