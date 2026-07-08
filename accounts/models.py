from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    face_image = models.ImageField(upload_to='face_images/')
    face_encoding = models.JSONField(null=True, blank=True)  # Stores 128-dimensional list of floats if face_recognition is used
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"
