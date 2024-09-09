from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='画像を選択')