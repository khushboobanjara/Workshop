"""Unit tests for conservative database-only knowledge matching."""

import unittest
from unittest.mock import MagicMock, patch

from src.database.knowledge_repository import (
    meaningful_tokens,
    score_knowledge_row,
    search_clinic_knowledge,
)


class ClinicKnowledgeMatchingTests(unittest.TestCase):
    def test_query_tokens_remove_common_question_words(self):
        self.assertEqual(
            meaningful_tokens("What is Machine Learning?"),
            {"machine", "learning"},
        )

    def test_exact_title_match_is_relevant(self):
        row = {
            "title": "Machine Learning",
            "keywords": "model training, AI",
            "answer": "Reviewed information about models and training.",
        }
        score, relevant = score_knowledge_row("What is Machine Learning?", row)
        self.assertTrue(relevant)
        self.assertGreaterEqual(score, 4.0)

    def test_unrelated_row_is_not_relevant(self):
        row = {
            "title": "Appointment booking",
            "keywords": "doctor, time slot",
            "answer": "Choose a doctor and available appointment time.",
        }
        score, relevant = score_knowledge_row("What is Machine Learning?", row)
        self.assertFalse(relevant)
        self.assertEqual(score, 0.0)

    def test_body_only_match_requires_strong_overlap(self):
        query = "explain symptoms of dengue fever"
        unrelated = {
            "title": "General clinic FAQ",
            "keywords": "clinic process",
            "answer": "This page explains how to use the clinic app.",
        }
        _, relevant = score_knowledge_row(query, unrelated)
        self.assertFalse(relevant)

    def test_search_returns_only_relevant_published_candidates(self):
        matching = {
            "id": 12,
            "title": "Machine Learning",
            "answer": "Stored database answer.",
            "keywords": "machine learning",
            "category": "HEALTH",
        }
        unrelated = {
            "id": 13,
            "title": "Clinic appointments",
            "answer": "Book a time with an available doctor.",
            "keywords": "doctor appointment",
            "category": "CLINIC",
        }
        cursor = MagicMock()
        cursor.fetchall.return_value = [unrelated, matching]
        connection = MagicMock()
        connection.cursor.return_value = cursor

        with patch(
            "src.database.knowledge_repository._get_db_connection",
            return_value=connection,
        ):
            result = search_clinic_knowledge("What is Machine Learning?")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], 12)
        self.assertEqual(result[0]["answer"], "Stored database answer.")
        cursor.execute.assert_called_once()
        connection.close.assert_called_once()
        cursor.close.assert_called_once()

    def test_unapproved_general_category_is_not_returned_even_if_it_matches(self):
        unrelated_domain = {
            "id": 44,
            "title": "Machine Learning",
            "answer": "An unrelated generic answer.",
            "keywords": "machine learning",
            "category": "GENERAL",
        }
        cursor = MagicMock()
        cursor.fetchall.return_value = [unrelated_domain]
        connection = MagicMock()
        connection.cursor.return_value = cursor

        with patch(
            "src.database.knowledge_repository._get_db_connection",
            return_value=connection,
        ):
            result = search_clinic_knowledge("What is Machine Learning?")

        self.assertEqual(result, [])

    def test_database_errors_propagate_for_fail_closed_service_response(self):
        with patch(
            "src.database.knowledge_repository._get_db_connection",
            side_effect=RuntimeError("database offline"),
        ):
            with self.assertRaises(RuntimeError):
                search_clinic_knowledge("clinic policy")


if __name__ == "__main__":
    unittest.main()
