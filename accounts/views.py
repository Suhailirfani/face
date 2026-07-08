import base64
import json
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.contrib import messages

from accounts.models import UserProfile
from accounts.face_utils import decode_image, detect_and_extract_face, verify_face_present, authenticate_by_face, HAS_FACE_RECOGNITION

def landing_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'accounts/landing.html')

def register_view(request):

    if request.user.is_authenticated:
        return redirect('dashboard')
        
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        face_image_file = request.FILES.get('face_image')
        face_image_base64 = request.POST.get('face_image_base64') # Captured from webcam
        
        if not username or not password:
            messages.error(request, "Username and password are required.")
            return render(request, 'accounts/register.html')
            
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return render(request, 'accounts/register.html')
            
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username is already taken.")
            return render(request, 'accounts/register.html')
            
        # Determine the image data
        image_data = None
        if face_image_base64:
            image_data = face_image_base64
        elif face_image_file:
            image_data = face_image_file
            
        if not image_data:
            messages.error(request, "Please upload a photo or capture your face using the webcam.")
            return render(request, 'accounts/register.html')
            
        # Decode and process face
        img_np = decode_image(image_data)
        if img_np is None:
            messages.error(request, "Failed to process image. Make sure it is a valid format.")
            return render(request, 'accounts/register.html')
            
        if not verify_face_present(img_np):
            messages.error(request, "No face detected in the image. Please position your face clearly in the camera.")
            return render(request, 'accounts/register.html')
            
        # Extract face encoding if using face_recognition
        encoding = None
        if HAS_FACE_RECOGNITION:
            encoding = detect_and_extract_face(img_np)
            if not encoding:
                messages.error(request, "Face detected but could not be processed for encoding. Try another photo.")
                return render(request, 'accounts/register.html')
                
        # Create User
        user = User.objects.create_user(username=username, password=password)
        
        # Save profile image file
        profile = UserProfile(user=user)
        if face_image_base64:
            # Convert base64 to django ContentFile
            try:
                format, imgstr = face_image_base64.split(';base64,')
                ext = format.split('/')[-1]
                img_data = base64.b64decode(imgstr)
                profile.face_image.save(f"{username}_face.{ext}", ContentFile(img_data), save=False)
            except Exception as e:
                user.delete()
                messages.error(request, f"Error saving captured image: {e}")
                return render(request, 'accounts/register.html')
        else:
            profile.face_image = face_image_file
            
        # Save encoding
        if encoding:
            profile.face_encoding = encoding
            
        profile.save()
        
        # Log the user in and redirect
        login(request, user)
        messages.success(request, f"Account created successfully! Logged in as {username}.")
        return redirect('dashboard')
        
    return render(request, 'accounts/register.html')

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
        
    if request.method == 'POST':
        # Traditional password login
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f"Logged in successfully as {username}!")
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password.")
            
    return render(request, 'accounts/login.html', {
        'has_face_recognition': HAS_FACE_RECOGNITION
    })

@csrf_exempt
def ajax_face_login(request):
    """
    AJAX endpoint for face recognition login. Receives base64 image data from front-end camera stream.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'Invalid request method.'}, status=400)
        
    try:
        data = json.loads(request.body)
        image_base64 = data.get('image')
        if not image_base64:
            return JsonResponse({'success': False, 'message': 'No image data provided.'}, status=400)
            
        # Authenticate face
        user, message = authenticate_by_face(image_base64)
        if user:
            login(request, user)
            return JsonResponse({'success': True, 'message': message, 'username': user.username})
        else:
            return JsonResponse({'success': False, 'message': message})
            
    except Exception as e:
        return JsonResponse({'success': False, 'message': f"Server error: {str(e)}"}, status=500)

@login_required
def dashboard_view(request):
    profile = getattr(request.user, 'profile', None)
    return render(request, 'accounts/dashboard.html', {
        'profile': profile,
        'has_face_recognition': HAS_FACE_RECOGNITION
    })

def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('login')
