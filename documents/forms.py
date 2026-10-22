from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

PRIVACY_NOTICE = (
    "I understand uploaded files are private to my account, but relevant excerpts "
    "from them and my question are sent to the configured AI provider to generate "
    "an answer. I will not use this service for emergencies or as a substitute "
    "for advice from my doctor."
)


class RegistrationForm(UserCreationForm):
    username = forms.EmailField(label="Email address", max_length=150)
    consent = forms.BooleanField(label=PRIVACY_NOTICE, required=True)

    class Meta:
        model = User
        fields = ("username",)

    def clean_username(self) -> str:
        return self.cleaned_data["username"].strip().casefold()


class EmailAuthenticationForm(AuthenticationForm):
    username = forms.EmailField(label="Email address", max_length=150)

    def clean(self):
        if self.cleaned_data.get("username"):
            self.cleaned_data["username"] = self.cleaned_data["username"].strip().casefold()
        return super().clean()
