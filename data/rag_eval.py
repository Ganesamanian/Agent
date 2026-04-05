DEFAULT_RETRIEVAL_EVAL_SET = [
    {
        "query": "customer cancelled inside free cancellation window but was still charged",
        "expected_docs": ["refunds_policy.md", "escalation_matrix.md"],
    },
    {
        "query": "property marked booking as no-show after guest says they cancelled",
        "expected_docs": ["refunds_policy.md", "payment_disputes.md"],
    },
    {
        "query": "chargeback dispute for duplicate property charge",
        "expected_docs": ["payment_disputes.md", "escalation_matrix.md"],
    },
    {
        "query": "customer requests deletion of personal data",
        "expected_docs": ["privacy_requests.md", "escalation_matrix.md"],
    },
]