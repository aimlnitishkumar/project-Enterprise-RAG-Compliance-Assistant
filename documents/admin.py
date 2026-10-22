from django.contrib import admin

from documents.models import UserDocument, UserProfile


@admin.register(UserDocument)
class UserDocumentAdmin(admin.ModelAdmin):
    list_display = ("original_name", "owner", "created_at")
    list_filter = ("created_at",)
    search_fields = ("original_name", "owner__username", "owner__email")
    readonly_fields = ("owner", "original_name", "file", "created_at")


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "remote_processing_consent_at", "consent_version")
    readonly_fields = ("user", "remote_processing_consent_at", "consent_version")
