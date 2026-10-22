from __future__ import annotations

from uuid import uuid4

from django.conf import settings
from django.core.validators import FileExtensionValidator
from django.db import models


def user_document_path(instance: "UserDocument", _filename: str) -> str:
    return f"user-documents/{instance.owner_id}/{uuid4().hex}.pdf"


class UserProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    remote_processing_consent_at = models.DateTimeField()
    consent_version = models.CharField(max_length=20, default="1")


class UserDocument(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="documents",
    )
    original_name = models.CharField(max_length=255)
    file = models.FileField(
        upload_to=user_document_path,
        validators=[FileExtensionValidator(allowed_extensions=["pdf"])],
        max_length=500,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.original_name
