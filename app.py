import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, random

# Функция для загрузки изображения
def load_image(image_path, max_dim=512):
    img = Image.open(image_path)
    img = img.convert("RGB")  # Убедимся, что изображение в формате RGB
    img = np.array(img)
    img = tf.image.resize(img, (max_dim, max_dim))  # Изменение размера
    img = img / 255.0  # Нормализация в диапазон [0, 1]
    return img

# Функция для случайного выбора изображения из папки
def random_image_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    random_image = random.choice(image_files)
    return os.path.join(folder_path, random_image)

# Функция для вычисления лосса
def compute_loss(model, target_image, content_image, style_image):
    target_outputs = model(target_image)
    content_outputs = model(content_image)
    style_outputs = model(style_image)
    
    # Потеря контента
    content_loss = tf.reduce_mean((content_outputs[0] - target_outputs[0])**2)
    
    # Потеря стиля
    style_loss = 0
    for a, b in zip(style_outputs[1:], target_outputs[1:]):
        gram_a = tf.linalg.einsum('bijc,bijd->bcd', a, a)
        gram_b = tf.linalg.einsum('bijc,bijd->bcd', b, b)
        style_loss += tf.reduce_mean((gram_a - gram_b)**2)
    
    # Общая потеря
    loss = content_loss + 1e-4 * style_loss
    return loss

# Шаг оптимизации
@tf.function
def train_step(model, target_image, content_image, style_image, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, target_image, content_image, style_image)
    grad = tape.gradient(loss, target_image)
    optimizer.apply_gradients([(grad, target_image)])
    target_image.assign(tf.clip_by_value(target_image, 0.0, 1.0))
    return loss

from google.colab import drive
drive.mount('/content/drive')

content_folder = '/content/drive/My Drive/datasets/impressionist/training/training'
style_folder = '/content/drive/My Drive/datasets/impressionist/validation/validation'

# Загружаем случайные изображения
content_image_path = random_image_from_folder(content_folder)
style_image_path = random_image_from_folder(style_folder)

# Загрузка изображений
content_image = load_image(content_image_path)
style_image = load_image(style_image_path)

# Визуализация выбранных изображений
st.image(content_image, caption="Content Image", use_column_width=True)
st.image(style_image, caption="Style Image", use_column_width=True)

# Загрузка модели (например, VGG19, без верхнего слоя)
def get_vgg_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    return vgg

# Создание модели
vgg_model = get_vgg_model()

# Ваши переменные для оптимизации
target_image = tf.Variable(content_image[tf.newaxis, ...], dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

# Обучение и отображение
epochs = 100
for epoch in range(epochs):
    loss = train_step(vgg_model, target_image, content_image[tf.newaxis, ...], style_image[tf.newaxis, ...], optimizer)
    
    # Показать результат
    if (epoch + 1) % 10 == 0:
        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        st.image(target_image.numpy()[0], caption=f"Stylized Image at Epoch {epoch+1}", use_column_width=True)

# Сохранение результата
output_image = target_image.numpy()[0]
output_image = np.clip(output_image, 0.0, 1.0)
output_image = Image.fromarray((output_image * 255).astype(np.uint8))
output_image.save('output_stylized_image.jpg')

# Показать результат
st.image('output_stylized_image.jpg', caption="Final Stylized Image", use_column_width=True)
