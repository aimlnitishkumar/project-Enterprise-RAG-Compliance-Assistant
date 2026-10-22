from __future__ import annotations

import logging

from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db import transaction
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from documents.api import (
    DocumentDownloadAPIView,
)
from documents.forms import EmailAuthenticationForm, RegistrationForm
from documents.models import UserDocument, UserProfile
from documents.serializers import DocumentUploadSerializer
from documents.services import delete_user_index, rebuild_user_index

logger = logging.getLogger(__name__)


def home(request: HttpRequest) -> HttpResponse:
    return redirect("dashboard" if request.user.is_authenticated else "login")


def register(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("dashboard")
    form = RegistrationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            user = form.save(commit=False)
            user.email = user.username
            user.save()
            UserProfile.objects.create(
                user=user,
                remote_processing_consent_at=timezone.now(),
            )
        login(request, user, backend="django.contrib.auth.backends.ModelBackend")
        return redirect("dashboard")
    return render(request, "documents/register.html", {"form": form})


def login_view(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("dashboard")
    form = EmailAuthenticationForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.get_user())
        next_url = request.GET.get("next", "")
        if url_has_allowed_host_and_scheme(
            next_url,
            allowed_hosts={request.get_host()},
            require_https=request.is_secure(),
        ):
            return redirect(next_url)
        return redirect("dashboard")
    return render(request, "documents/login.html", {"form": form})


@require_POST
def logout_view(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        logout(request)
    return redirect("login")


@login_required
def dashboard(request: HttpRequest) -> HttpResponse:
    documents = UserDocument.objects.filter(owner=request.user)
    return render(
        request,
        "documents/dashboard.html",
        {"documents": documents, "email": request.user.email or request.user.username},
    )


@login_required
@require_POST
def upload_document(request: HttpRequest) -> HttpResponse:
    serializer = DocumentUploadSerializer(
        data=request.FILES,
        context={"request": request},
    )
    if not serializer.is_valid():
        messages.error(request, "; ".join(serializer.errors.get("file", ["Invalid PDF."])))
        return redirect("dashboard")
    document = serializer.save()
    indexed = False
    try:
        chunks = rebuild_user_index(request.user.pk)
        indexed = True
    except (OSError, RuntimeError, ValueError) as exc:
        logger.exception("Failed to index an uploaded document")
        messages.error(request, f"Could not index this PDF: {exc}")
        return redirect("dashboard")
    finally:
        if not indexed and document.pk:
            document.delete()
    messages.success(request, f"PDF indexed successfully ({chunks} text chunks).")
    return redirect("dashboard")


@login_required
def download_document(request: HttpRequest, document_id: int) -> HttpResponse:
    return DocumentDownloadAPIView.as_view()(request, document_id=document_id)


@login_required
@require_POST
def delete_document(request: HttpRequest, document_id: int) -> HttpResponse:
    try:
        document = UserDocument.objects.get(pk=document_id, owner=request.user)
    except UserDocument.DoesNotExist as exc:
        raise Http404 from exc
    remaining_documents = list(
        UserDocument.objects.filter(owner=request.user).exclude(pk=document.pk).order_by("pk")
    )
    try:
        rebuild_user_index(request.user.pk, remaining_documents)
    except Exception:
        logger.exception("Failed to refresh the private document index")
        messages.error(request, "Could not refresh the document index. The PDF was not deleted.")
        return redirect("dashboard")
    document.delete()
    messages.success(request, "PDF and its indexed passages were deleted.")
    return redirect("dashboard")


@login_required
@require_POST
def delete_account(request: HttpRequest) -> HttpResponse:
    if not request.user.check_password(request.POST.get("password", "")):
        messages.error(request, "Enter your current password to delete your account.")
        return redirect("dashboard")
    user: User = request.user
    try:
        delete_user_index(user.pk)
    except Exception:
        logger.exception("Failed to remove a user's vector collection")
        messages.error(request, "Could not remove the document index.")
        return redirect("dashboard")
    logout(request)
    user.delete()
    messages.success(request, "Your account and uploaded PDFs were deleted.")
    return redirect("login")
