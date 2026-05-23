import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Force test environment variables before anything is imported
os.environ["ALLOW_DEGRADED_STARTUP"] = "1"
os.environ["SUPABASE_URL"] = "https://mock.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "mockservicekey"
os.environ["SLA_ESCALATION_ENABLED"] = "false"

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class FakeResult:
    def __init__(self, data=None):
        self.data = data or []


class FakeTable:
    def __init__(self, db, name):
        self.db = db
        self.name = name
        self.filters = {}
        self.payload = None
        self.limit_count = None
        self.is_single = False
        self.order_field = None
        self.order_desc = False

    def select(self, *_args):
        return self

    def update(self, payload):
        self.payload = payload
        return self

    def insert(self, payload):
        rows = payload if isinstance(payload, list) else [payload]
        for r in rows:
            if "id" not in r:
                r["id"] = len(self.db.get(self.name, [])) + 1
        self.db.setdefault(self.name, []).extend(rows)
        self.inserted_rows = rows
        return self

    def eq(self, field, value):
        self.filters[field] = value
        return self

    def order(self, field, desc=False):
        self.order_field = field
        self.order_desc = desc
        return self

    def limit(self, value):
        self.limit_count = value
        return self

    def single(self):
        self.is_single = True
        return self

    def execute(self):
        if hasattr(self, "inserted_rows"):
            return FakeResult(self.inserted_rows)

        if self.payload is not None:
            rows = self.db.setdefault(self.name, [])
            updated_rows = []
            for row in rows:
                match = True
                for k, v in self.filters.items():
                    if row.get(k) != v:
                        match = False
                        break
                if match:
                    row.update(self.payload)
                    updated_rows.append(row)
            return FakeResult(updated_rows)

        rows = list(self.db.get(self.name, []))
        for key, value in self.filters.items():
            rows = [row for row in rows if row.get(key) == value]

        if self.order_field:
            rows.sort(key=lambda x: x.get(self.order_field, ""), reverse=self.order_desc)

        if self.limit_count is not None:
            rows = rows[:self.limit_count]

        if self.is_single:
            return FakeResult(rows[0] if rows else None)
        return FakeResult(rows)


class FakeSupabase:
    def __init__(self, db):
        self.db = db

    def table(self, name):
        return FakeTable(self.db, name)

    def rpc(self, name, params):
        # Mock simple search matching logic
        query = params.get("query_text", "").lower()
        company_id = params.get("company_id")
        
        matching = []
        tickets = self.db.get("tickets", [])
        for t in tickets:
            if company_id and t.get("company_id") != company_id:
                continue
            if query in str(t.get("subject", "")).lower() or query in str(t.get("description", "")).lower():
                matching.append(t)
        
        class RpcMock:
            def execute(self):
                return FakeResult(matching)
        return RpcMock()


@pytest.fixture
def fake_db():
    return {
        "system_settings": [
            {
                "company_id": "company_A",
                "ai_confidence_threshold": 0.80,
                "duplicate_sensitivity": 0.85,
                "enable_auto_resolve": True
            },
            {
                "company_id": "company_B",
                "ai_confidence_threshold": 0.80,
                "duplicate_sensitivity": 0.85,
                "enable_auto_resolve": True
            }
        ],
        "profiles": [
            {
                "id": "user_A",
                "company_id": "company_A",
                "company": "Company A"
            },
            {
                "id": "user_B",
                "company_id": "company_B",
                "company": "Company B"
            }
        ],
        "tickets": [],
        "ticket_messages": [],
        "audit_logs": []
    }


@pytest.fixture
def fake_supabase(fake_db):
    return FakeSupabase(fake_db)


@pytest.fixture(autouse=True)
def mock_ai_services():
    with patch("backend.services.classifier_service.ClassifierService.predict") as mock_v1_predict, \
         patch("backend.services.classifier_service.ClassifierService.load") as mock_v1_load, \
         patch("backend.services.classifier_v3.ClassifierServiceV3.predict") as mock_v3_predict, \
         patch("backend.services.ner_service.NERService.extract_entities") as mock_ner_extract, \
         patch("backend.services.ner_service.NERService.load") as mock_ner_load, \
         patch("backend.services.duplicate_service.DuplicateService.find_semantic_duplicate") as mock_dup_find, \
         patch("backend.services.duplicate_service.DuplicateService.check_duplicate") as mock_dup_check, \
         patch("backend.services.duplicate_service.DuplicateService.generate_embedding") as mock_dup_emb, \
         patch("backend.services.duplicate_service.DuplicateService.load") as mock_dup_load, \
         patch("backend.services.rag_service.RagService.search_knowledge_base") as mock_rag_search, \
         patch("backend.services.rag_service.RagService.load") as mock_rag_load:

        # Stub default returns
        mock_v1_predict.return_value = {
            "category": "Software",
            "subcategory": "Software Install",
            "priority": "Medium",
            "auto_resolve": True,
            "assigned_team": "Application Support",
            "confidence": 0.95,
        }
        mock_v3_predict.return_value = {
            "Category": {"prediction": "Software", "confidence": 0.95},
            "Subcategory": {"prediction": "Software Install", "confidence": 0.95},
            "priority": {"prediction": "Medium", "confidence": 0.95},
        }
        mock_ner_extract.return_value = [
            {"text": "VPN", "label": "PRODUCT", "confidence": 0.99}
        ]
        mock_dup_find.return_value = {
            "is_duplicate": False,
            "duplicate_ticket_id": None,
            "parent_ticket_id": None,
            "is_potential_duplicate": False,
            "similarity": 0.0,
        }
        mock_dup_check.return_value = mock_dup_find.return_value
        mock_dup_emb.return_value = [0.1] * 384
        mock_rag_search.return_value = None

        yield


@pytest.fixture
def test_client(fake_supabase):
    with patch("backend.main.supabase", fake_supabase):
        from fastapi.testclient import TestClient
        from backend.main import app
        # Ensure ALLOW_DEGRADED_STARTUP is set during client initialization
        with patch.dict(os.environ, {"ALLOW_DEGRADED_STARTUP": "1"}):
            with TestClient(app) as client:
                yield client
