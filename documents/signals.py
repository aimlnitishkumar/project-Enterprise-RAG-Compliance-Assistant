from django.db.models.signals import post_delete
from django.dispatch import receiver

from documents.models import UserDocument


@receiver(post_delete, sender=UserDocument)
def delete_document_file(sender, instance: UserDocument, **kwargs) -> None:
    if instance.file:
        instance.file.delete(save=False)
