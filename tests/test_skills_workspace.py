import os
import tempfile
import pytest
import shutil
from services.memory.skills import SkillsManager
from src.agent_loop import _build_layered_instructions
from src.tool_implementations import do_manage_skills

@pytest.fixture
def temp_workspace():
    # Setup temporary directory structure representing a workspace
    ws = tempfile.mkdtemp()
    skills_dir = os.path.join(ws, ".agents", "skills", "general", "workspace_test_skill")
    os.makedirs(skills_dir, exist_ok=True)
    
    skill_content = """---
name: workspace_test_skill
description: A workspace test skill
category: general
status: published
---
## Procedure
1. Step one
"""
    with open(os.path.join(skills_dir, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(skill_content)
        
    # Also add AGENTS.md representing repo instructions
    with open(os.path.join(ws, "AGENTS.md"), "w", encoding="utf-8") as f:
        f.write("Workspace local AGENTS.md instruction content")
        
    yield ws
    shutil.rmtree(ws)

def test_skills_manager_workspace_load(temp_workspace):
    # Setup a standard SkillsManager in another temp dir
    data_dir = tempfile.mkdtemp()
    try:
        sm = SkillsManager(data_dir)
        
        # Load without workspace - should not see the workspace skill
        skills = sm.load(owner="test_user")
        assert not any(s.get("name") == "workspace-test-skill" for s in skills)
        
        # Load with workspace - should see it as read-only
        skills_ws = sm.load(owner="test_user", workspace=temp_workspace)
        match = next((s for s in skills_ws if s.get("name") == "workspace-test-skill"), None)
        assert match is not None
        assert match.get("read_only") is True
        
        # read_skill_md should find it when workspace is provided
        md = sm.read_skill_md("workspace-test-skill", owner="test_user", workspace=temp_workspace)
        assert md is not None
        assert "workspace_test_skill" in md
    finally:
        shutil.rmtree(data_dir)

@pytest.mark.asyncio
async def test_do_manage_skills_read_only_blocks(temp_workspace):
    data_dir = tempfile.mkdtemp()
    try:
        # Verify that manage_skills blocks edit/delete on the workspace skill
        # Because do_manage_skills instantiates SkillsManager internally, we must ensure it picks up DATA_DIR
        from unittest.mock import patch
        with patch("src.constants.DATA_DIR", data_dir):
            import json
            # Try to edit the read-only skill
            res = await do_manage_skills(
                json.dumps({"action": "edit", "name": "workspace-test-skill", "content": "---\nname: workspace_test_skill\n---"}),
                owner="test_user",
                workspace=temp_workspace
            )
            assert res.get("exit_code") == 1
            assert "read-only" in res.get("error", "")
            
            # Try to delete it
            res_del = await do_manage_skills(
                json.dumps({"action": "delete", "name": "workspace-test-skill"}),
                owner="test_user",
                workspace=temp_workspace
            )
            assert res_del.get("exit_code") == 1
            assert "read-only" in res_del.get("error", "")
    finally:
        shutil.rmtree(data_dir)

def test_layered_instructions_use_workspace(temp_workspace):
    # Set preference / settings toggles so repo instructions are enabled
    from routes.prefs_routes import _load_for_user, _save_for_user
    prefs = _load_for_user("test_user")
    prefs["agent_context_repo_instructions_enabled"] = True
    _save_for_user("test_user", prefs)
    
    # Generate layered instructions without workspace (should not find the workspace AGENTS.md)
    inst_no_ws = _build_layered_instructions(owner="test_user")
    assert "Workspace local AGENTS.md instruction content" not in inst_no_ws
    
    # Generate layered instructions with workspace (should load AGENTS.md from workspace)
    inst_ws = _build_layered_instructions(owner="test_user", workspace=temp_workspace)
    assert "Workspace local AGENTS.md" in inst_ws

@pytest.mark.asyncio
async def test_invoke_skill_route_uses_workspace(temp_workspace):
    from routes.skills_routes import setup_skills_routes
    from fastapi import Request
    from fastapi.datastructures import State
    from fastapi import HTTPException
    import json

    def _make_request(user: str, body=None) -> Request:
        class DummyApp:
            state = State()
        payload = json.dumps(body).encode("utf-8") if body is not None else b""
        sent = False
        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        return Request(scope={
            "type": "http",
            "method": "POST",
            "headers": [(b"content-type", b"application/json")],
            "app": DummyApp(),
            "state": {"current_user": user},
        }, receive=receive)

    data_dir = tempfile.mkdtemp()
    try:
        sm = SkillsManager(data_dir)
        router = setup_skills_routes(sm)
        
        # Find the invoke route endpoint
        invoke_endpoint = next(
            route.endpoint for route in router.routes
            if route.path == "/api/skills/{skill_id}/invoke" and "POST" in route.methods
        )
        
        # Without workspace: should raise 404 (not found)
        with pytest.raises(HTTPException) as exc:
            await invoke_endpoint(
                _make_request("test_user", {"request": "hello"}),
                "workspace-test-skill",
                workspace=None
            )
        assert exc.value.status_code == 404
        
        # With workspace: should succeed
        res = await invoke_endpoint(
            _make_request("test_user", {"request": "hello"}),
            "workspace-test-skill",
            workspace=temp_workspace
        )
        assert res.get("ok") is True
        assert res.get("name") == "workspace-test-skill"
        assert "Workspace local AGENTS.md" not in res.get("message")
        assert "Step one" in res.get("message")
    finally:
        shutil.rmtree(data_dir)

