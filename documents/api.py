from __future__ import annotations

import logging

from django.contrib.auth import authenticate
from django.db import transaction
from django.http import FileResponse, Http404
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken

from documents.models import UserDocument, UserProfile
from documents.serializers import (
    DocumentUploadSerializer,
    EmailTokenSerializer,
    LogoutSerializer,
    QuestionSerializer,
    RegistrationSerializer,
    UserDocumentSerializer,
)
from documents.services import answer_question, delete_user_index, rebuild_user_index

logger = logging.getLogger(__name__)


class RegisterAPIView(APIView):
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "register"

    def post(self, request):
        serializer = RegistrationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {"id": user.pk, "email": user.email},
            status=status.HTTP_201_CREATED,
        )


class EmailTokenAPIView(APIView):
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "token"

    def post(self, request):
        serializer = EmailTokenSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        return Response(serializer.validated_data)


class DocumentListCreateAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "upload"

    def get(self, request):
        documents = UserDocument.objects.filter(owner=request.user)
        serializer = UserDocumentSerializer(
            documents,
            many=True,
            context={"request": request},
        )
        return Response(serializer.data)

    def post(self, request):
        serializer = DocumentUploadSerializer(
            data=request.data,
            context={"request": request},
        )
        serializer.is_valid(raise_exception=True)
        document = serializer.save()
        indexed = False
        try:
            chunks = rebuild_user_index(request.user.pk)
            indexed = True
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Failed to index an uploaded document")
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        finally:
            if not indexed and document.pk:
                document.delete()
        return Response(
            {
                **UserDocumentSerializer(
                    document,
                    context={"request": request},
                ).data,
                "indexed_chunks": chunks,
            },
            status=status.HTTP_201_CREATED,
        )


class DocumentDetailAPIView(APIView):
    def delete(self, request, document_id: int):
        try:
            document = UserDocument.objects.get(pk=document_id, owner=request.user)
        except UserDocument.DoesNotExist as exc:
            raise Http404 from exc
        remaining_documents = list(
            UserDocument.objects.filter(owner=request.user)
            .exclude(pk=document.pk)
            .order_by("pk")
        )
        try:
            rebuild_user_index(request.user.pk, remaining_documents)
        except Exception:
            logger.exception("Failed to refresh the private document index")
            return Response(
                {"detail": "Could not update the document index. The file was not deleted."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        document.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class DocumentDownloadAPIView(APIView):
    def get(self, request, document_id: int):
        try:
            document = UserDocument.objects.get(pk=document_id, owner=request.user)
            file_handle = document.file.open("rb")
        except (UserDocument.DoesNotExist, OSError, ValueError) as exc:
            raise Http404 from exc
        return FileResponse(file_handle, as_attachment=True, filename=document.original_name)


class ChatAPIView(APIView):
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "chat"

    def post(self, request):
        if not UserProfile.objects.filter(
            user=request.user,
            remote_processing_consent_at__isnull=False,
        ).exists():
            return Response(
                {"detail": "Accept the privacy notice before asking a question."},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = QuestionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = answer_question(
            request.user.pk,
            serializer.validated_data["question"],
            top_k=serializer.validated_data["top_k"],
        )
        return Response(result)


class LogoutAPIView(APIView):
    def post(self, request):
        serializer = LogoutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            refresh = RefreshToken(serializer.validated_data["refresh"])
            if str(refresh["user_id"]) != str(request.user.pk):
                return Response(
                    {"detail": "The refresh token does not belong to this account."},
                    status=status.HTTP_403_FORBIDDEN,
                )
            refresh.blacklist()
        except TokenError:
            return Response(
                {"detail": "The refresh token is invalid or expired."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response(status=status.HTTP_204_NO_CONTENT)


class AccountAPIView(APIView):
    def get(self, request):
        return Response(
            {
                "id": request.user.pk,
                "email": request.user.email,
                "document_count": UserDocument.objects.filter(owner=request.user).count(),
            }
        )

    def delete(self, request):
        password = request.data.get("password", "")
        if not isinstance(password, str) or not authenticate(
            request=request,
            username=request.user.username,
            password=password,
        ):
            return Response(
                {"detail": "Enter your current password to delete this account."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        user = request.user
        user_id = user.pk
        try:
            delete_user_index(user_id)
        except Exception:
            logger.exception("Failed to remove a user's vector collection")
            return Response(
                {"detail": "Could not remove this account's document index."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        with transaction.atomic():
            user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
