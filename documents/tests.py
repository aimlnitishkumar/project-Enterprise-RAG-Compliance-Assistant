from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from documents.models import UserDocument, UserProfile
from documents.services import user_collection_name

VALID_PDF = b"%PDF-1.4\n% test PDF upload\n"
CONSENT = "I understand uploaded files are private to my account"


class AccountWebTests(TestCase):
    def test_registration_requires_and_records_privacy_consent(self):
        payload = {
            "username": "Researcher@example.com",
            "password1": "A-long-unique-password-2048!",
            "password2": "A-long-unique-password-2048!",
        }
        response = self.client.post("/accounts/register/", payload)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.exists())

        payload["consent"] = "on"
        response = self.client.post("/accounts/register/", payload)
        self.assertRedirects(response, "/dashboard/")
        user = User.objects.get(username="researcher@example.com")
        self.assertTrue(user.check_password(payload["password1"]))
        self.assertEqual(user.email, "researcher@example.com")
        self.assertTrue(user.profile.remote_processing_consent_at)

    def test_login_uses_email_and_does_not_allow_external_redirects(self):
        user = User.objects.create_user(
            username="reader@example.com",
            email="reader@example.com",
            password="A-long-unique-password-2048!",
        )
        UserProfile.objects.create(
            user=user,
            remote_processing_consent_at=datetime.now(timezone.utc),
        )
        response = self.client.post(
            "/accounts/login/?next=https://example.org/",
            {
                "username": "READER@example.com",
                "password": "A-long-unique-password-2048!",
            },
        )
        self.assertRedirects(response, "/dashboard/")
        self.assertEqual(self.client.get("/dashboard/").status_code, 200)


class AccountApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="owner@example.com",
            email="owner@example.com",
            password="A-long-unique-password-2048!",
        )
        UserProfile.objects.create(
            user=self.user,
            remote_processing_consent_at=datetime.now(timezone.utc),
        )

    def test_registration_requires_privacy_consent_and_valid_password(self):
        response = self.client.post(
            "/api/auth/register/",
            {"email": "new@example.com", "password": "A-long-unique-password-2048!"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(User.objects.filter(username="new@example.com").exists())

        response = self.client.post(
            "/api/auth/register/",
            {
                "email": "New@Example.com",
                "password": "A-long-unique-password-2048!",
                "consent": True,
            },
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        created_user = User.objects.get(username="new@example.com")
        self.assertTrue(created_user.profile.remote_processing_consent_at)

    def test_registration_rejects_passwords_based_on_the_email(self):
        response = self.client.post(
            "/api/auth/register/",
            {
                "email": "researcher@example.com",
                "password": "researcher@example.com-Strong-2048!",
                "consent": True,
            },
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(User.objects.filter(username="researcher@example.com").exists())

    def test_jwt_sign_in_and_authenticated_account_api(self):
        response = self.client.post(
            "/api/auth/token/",
            {
                "email": "OWNER@example.com",
                "password": "A-long-unique-password-2048!",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access", response.data)
        self.assertIn("refresh", response.data)

        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
        account = self.client.get("/api/account/")
        self.assertEqual(account.status_code, 200)
        self.assertEqual(account.data["email"], self.user.email)

    def test_anonymous_users_cannot_read_the_api(self):
        response = self.client.get("/api/documents/")
        self.assertEqual(response.status_code, 401)

    def test_session_authenticated_api_requires_csrf(self):
        client = APIClient(enforce_csrf_checks=True)
        client.force_login(self.user)
        response = client.post(
            "/api/chat/",
            {"question": "What does my document say?"},
            format="json",
        )
        self.assertEqual(response.status_code, 403)

    @override_settings(MAX_PDF_UPLOAD_BYTES=1024)
    def test_document_upload_requires_a_pdf_and_is_owner_scoped(self):
        other_user = User.objects.create_user(
            username="other@example.com",
            password="A-long-unique-password-2048!",
        )
        with TemporaryDirectory() as media_directory, override_settings(
            MEDIA_ROOT=media_directory
        ):
            self.client.force_authenticate(self.user)
            with patch("documents.api.rebuild_user_index", return_value=3) as rebuild:
                response = self.client.post(
                    "/api/documents/",
                    {"file": ContentFile(VALID_PDF, name="medical-record.pdf")},
                    format="multipart",
                )
            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.data["indexed_chunks"], 3)
            rebuild.assert_called_once_with(self.user.pk)
            document = UserDocument.objects.get(owner=self.user)
            self.assertTrue(Path(document.file.path).is_file())
            self.assertIn(str(self.user.pk), document.file.name)

            other_user_document = UserDocument.objects.create(
                owner=other_user,
                original_name="private.pdf",
                file=ContentFile(VALID_PDF, name="private.pdf"),
            )
            self.assertEqual(
                list(UserDocument.objects.filter(owner=self.user)),
                [document],
            )

            self.client.force_authenticate(other_user)
            other_documents = self.client.get("/api/documents/")
            self.assertEqual(other_documents.status_code, 200)
            self.assertEqual(
                [item["id"] for item in other_documents.data],
                [other_user_document.pk],
            )
            forbidden_download = self.client.get(
                f"/api/documents/{document.pk}/download/"
            )
            self.assertEqual(forbidden_download.status_code, 404)
            forbidden_delete = self.client.delete(f"/api/documents/{document.pk}/")
            self.assertEqual(forbidden_delete.status_code, 404)
            self.assertTrue(UserDocument.objects.filter(pk=document.pk).exists())

    def test_upload_rejects_non_pdf_content(self):
        self.client.force_authenticate(self.user)
        response = self.client.post(
            "/api/documents/",
            {"file": ContentFile(b"not a PDF", name="medical-record.pdf")},
            format="multipart",
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(UserDocument.objects.exists())

    def test_failed_document_indexing_removes_the_uploaded_file_and_record(self):
        self.client.force_authenticate(self.user)
        with TemporaryDirectory() as media_directory, override_settings(
            MEDIA_ROOT=media_directory
        ):
            with patch(
                "documents.api.rebuild_user_index",
                side_effect=RuntimeError("Tesseract OCR is unavailable."),
            ):
                response = self.client.post(
                    "/api/documents/",
                    {"file": ContentFile(VALID_PDF, name="medical-record.pdf")},
                    format="multipart",
                )
            self.assertEqual(response.status_code, 422)
            self.assertFalse(UserDocument.objects.exists())
            self.assertEqual(list(Path(media_directory).rglob("*.pdf")), [])

    def test_chroma_collection_names_are_unique_per_account(self):
        other_user = User.objects.create_user(
            username="other@example.com",
            password="A-long-unique-password-2048!",
        )
        self.assertNotEqual(
            user_collection_name(self.user.pk),
            user_collection_name(other_user.pk),
        )
        self.assertEqual(
            user_collection_name(self.user.pk),
            f"shryai_user_{self.user.pk}",
        )

    def test_index_rebuild_uses_only_the_selected_users_collection(self):
        from documents.services import rebuild_user_index

        with TemporaryDirectory() as media_directory, override_settings(
            MEDIA_ROOT=media_directory
        ):
            document = UserDocument.objects.create(
                owner=self.user,
                original_name="doctor-report.pdf",
                file=ContentFile(VALID_PDF, name="doctor-report.pdf"),
            )
            with (
                patch("documents.services._embeddings", return_value=object()),
                patch(
                    "documents.services.rebuild_vector_store",
                    return_value=(None, 1),
                ) as rebuild,
            ):
                self.assertEqual(rebuild_user_index(self.user.pk, [document]), 1)
            self.assertEqual(
                rebuild.call_args.kwargs["collection_name"],
                user_collection_name(self.user.pk),
            )
            self.assertEqual(
                rebuild.call_args.kwargs["source_names"],
                {str(Path(document.file.path).resolve()): "doctor-report.pdf"},
            )

    def test_account_deletion_requires_password_and_purges_private_index(self):
        self.client.force_authenticate(self.user)
        user_id = self.user.pk
        with patch("documents.api.delete_user_index") as delete_index:
            response = self.client.delete(
                "/api/account/",
                {"password": "wrong-password"},
                format="json",
            )
            self.assertEqual(response.status_code, 400)
            delete_index.assert_not_called()

            response = self.client.delete(
                "/api/account/",
                {"password": "A-long-unique-password-2048!"},
                format="json",
            )
        self.assertEqual(response.status_code, 204)
        delete_index.assert_called_once_with(user_id)
        self.assertFalse(User.objects.filter(pk=user_id).exists())

    def test_chat_requires_saved_privacy_consent(self):
        UserProfile.objects.filter(user=self.user).delete()
        self.client.force_authenticate(self.user)
        response = self.client.post(
            "/api/chat/",
            {"question": "What does this health report say?", "top_k": 4},
            format="json",
        )
        self.assertEqual(response.status_code, 403)
