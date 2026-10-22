from django.contrib import admin
from django.urls import include, path

from documents import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.home, name="home"),
    path("accounts/register/", views.register, name="register"),
    path("accounts/login/", views.login_view, name="login"),
    path("accounts/logout/", views.logout_view, name="logout"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("documents/upload/", views.upload_document, name="document-upload"),
    path("documents/<int:document_id>/download/", views.download_document, name="document-download"),
    path("documents/<int:document_id>/delete/", views.delete_document, name="document-delete"),
    path("accounts/delete/", views.delete_account, name="account-delete"),
    path("api/", include("documents.api_urls")),
]
