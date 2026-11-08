"""Shared fixtures for agent tests.  No network, no FAISS, no embedding model."""
from __future__ import annotations

from datetime import date

import pytest

from backend.agents.fake_retriever import KeywordRetriever
from backend.agents.state import DisputeInput

# The brief's section 4.1 case: SMS alert the same day, reported three weeks later,
# OTP shared earlier with a caller impersonating the bank.
NARRATIVE_41 = (
    "The bank's SMS alert for the transaction came the same day, but I did not notice it and reported it "
    "three weeks later. Before that, I had shared an OTP with someone who called saying he was from the bank."
)


@pytest.fixture(scope="session")
def retriever() -> KeywordRetriever:
    return KeywordRetriever()


@pytest.fixture
def request_41() -> DisputeInput:
    return DisputeInput(
        narrative=NARRATIVE_41, channel="card_not_present", amount=48000.0,
        transaction_date=date(2026, 10, 1), communication_date=date(2026, 10, 1),
        reported_date=date(2026, 10, 22), merchant="ONLINE ELECTRONICS STORE", account_type="savings",
    )


@pytest.fixture
def request_ok() -> DisputeInput:
    return DisputeInput(
        narrative="A card-not-present charge I never made appeared on my card. I never shared any details "
                  "and reported it the next day after the SMS alert.",
        channel="card_not_present", amount=4999.0,
        transaction_date=date(2026, 10, 5), communication_date=date(2026, 10, 5),
        reported_date=date(2026, 10, 6), merchant="UNKNOWN WEB MERCHANT", account_type="savings",
    )
