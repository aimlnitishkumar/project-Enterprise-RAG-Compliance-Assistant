from __future__ import annotations

from pathlib import PurePosixPath

from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework import serializers
from rest_framework.reverse import reverse
from rest_framework_simplejwt.tokens import RefreshToken

from documents.forms import PRIVACY_NOTICE
from documents.models import UserDocument, UserProfile


class RegistrationSerializer(serializers.Serializer):
    email = serializers.EmailField(max_length=150)
    password = serializers.CharField(write_only=True, trim_whitespace=False)
    consent = serializers.BooleanField(required=True)

    def validate_email(self, value: str) -> str:
        value = value.strip().casefold()
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("An account with this email already exists.")
        return value

    def validate_password(self, value: str) -> str:
        candidate_user = User(username=self.initial_data.get("email", ""))
        try:
            validate_password(value, user=candidate_user)
        except DjangoValidationError as exc:
            raise serializers.ValidationError(list(exc.messages)) from exc
        return value

    def validate_consent(self, value: bool) -> bool:
        if not value:
            raise serializers.ValidationError(PRIVACY_NOTICE)
        return value

    def create(self, validated_data: dict) -> User:
        from django.utils import timezone

        email = validated_data["email"]
        user = User.objects.create_user(
            username=email,
            email=email,
            password=validated_data["password"],
        )
        UserProfile.objects.create(
            user=user,
            remote_processing_consent_at=timezone.now(),
        )
        return user


class EmailTokenSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate(self, attrs: dict) -> dict:
        from django.contrib.auth import authenticate

        email = attrs["email"].strip().casefold()
        user = authenticate(
            request=self.context.get("request"),
            username=email,
            password=attrs["password"],
        )
        if user is None:
            raise serializers.ValidationError("Invalid email or password.")
        refresh = RefreshToken.for_user(user)
        return {
            "access": str(refresh.access_token),
            "refresh": str(refresh),
        }


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

    def validate_file(self, uploaded_file):
        if uploaded_file.size > settings.MAX_PDF_UPLOAD_BYTES:
            raise serializers.ValidationError("PDF files must be 20 MB or smaller.")
        original_name = uploaded_file.name.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not original_name.casefold().endswith(".pdf"):
            raise serializers.ValidationError("Only PDF files can be uploaded.")
        header = uploaded_file.read(1024)
        uploaded_file.seek(0)
        if b"%PDF-" not in header:
            raise serializers.ValidationError("The uploaded file is not a valid PDF.")
        uploaded_file.name = original_name
        return uploaded_file

    def create(self, validated_data: dict) -> UserDocument:
        uploaded_file = validated_data["file"]
        return UserDocument.objects.create(
            owner=self.context["request"].user,
            original_name=uploaded_file.name[:255],
            file=uploaded_file,
        )


class UserDocumentSerializer(serializers.ModelSerializer):
    download_url = serializers.SerializerMethodField()

    class Meta:
        model = UserDocument
        fields = ("id", "original_name", "created_at", "download_url")
        read_only_fields = fields

    def get_download_url(self, document: UserDocument) -> str:
        return reverse(
            "document-download",
            kwargs={"document_id": document.pk},
            request=self.context.get("request"),
        )


class QuestionSerializer(serializers.Serializer):
    question = serializers.CharField(min_length=2, max_length=2000, trim_whitespace=True)
    top_k = serializers.IntegerField(min_value=1, max_value=8, default=4)


class LogoutSerializer(serializers.Serializer):
    refresh = serializers.CharField()
