from django.urls import path
from rest_framework_simplejwt.views import TokenBlacklistView, TokenRefreshView

from documents.api import (
    AccountAPIView,
    ChatAPIView,
    DocumentDetailAPIView,
    DocumentDownloadAPIView,
    DocumentListCreateAPIView,
    EmailTokenAPIView,
    LogoutAPIView,
    RegisterAPIView,
)

urlpatterns = [
    path("auth/register/", RegisterAPIView.as_view(), name="api-register"),
    path("auth/token/", EmailTokenAPIView.as_view(), name="api-token"),
    path("auth/token/refresh/", TokenRefreshView.as_view(), name="api-token-refresh"),
    path("auth/token/blacklist/", TokenBlacklistView.as_view(), name="api-token-blacklist"),
    path("auth/logout/", LogoutAPIView.as_view(), name="api-logout"),
    path("account/", AccountAPIView.as_view(), name="api-account"),
    path("documents/", DocumentListCreateAPIView.as_view(), name="api-documents"),
    path(
        "documents/<int:document_id>/",
        DocumentDetailAPIView.as_view(),
        name="api-document-detail",
    ),
    path(
        "documents/<int:document_id>/download/",
        DocumentDownloadAPIView.as_view(),
        name="api-document-download",
    ),
    path("chat/", ChatAPIView.as_view(), name="api-chat"),
]
