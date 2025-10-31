#!/usr/bin/env python3
"""
CNN Model Architecture for Doodle Recognition
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from config import MODEL_CONFIG

class DoodleClassifier:
    """CNN model for doodle classification"""
    
    def __init__(self, num_classes, input_shape=(28, 28, 1)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
    
    def build_simple_cnn(self):
        """Build a simple CNN model"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third conv block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_advanced_cnn(self):
        """Build an advanced CNN with residual connections"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial conv layer
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual block 1
        residual = x
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Residual block 2
        residual = layers.Conv2D(64, (1, 1), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Residual block 3
        residual = layers.Conv2D(128, (1, 1), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def build_mobilenet_transfer(self):
        """Build model using MobileNetV2 transfer learning"""
        # Create base model
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),  # MobileNet expects 224x224x3
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add preprocessing for single channel to RGB
        inputs = layers.Input(shape=self.input_shape)
        
        # Resize to 224x224 and convert to RGB
        x = layers.UpSampling2D(size=(8, 8))(inputs)  # 28x28 -> 224x224
        x = layers.Conv2D(3, (1, 1), padding='same')(x)  # 1 channel -> 3 channels
        
        # Apply base model
        x = base_model(x, training=False)
        
        # Add custom head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def build_lightweight_cnn(self):
        """Build a lightweight CNN for fast inference"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Depthwise separable convolutions for efficiency
            layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_model(self, architecture='simple'):
        """Build model based on specified architecture"""
        print(f"🏗️  Building {architecture} model...")
        
        if architecture == 'simple':
            self.model = self.build_simple_cnn()
        elif architecture == 'advanced':
            self.model = self.build_advanced_cnn()
        elif architecture == 'mobilenet':
            self.model = self.build_mobilenet_transfer()
        elif architecture == 'lightweight':
            self.model = self.build_lightweight_cnn()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Print model summary
        print(f"✅ Model built successfully!")
        print(f"📊 Model parameters: {self.model.count_params():,}")
        
        return self.model
    
    def compile_model(self, optimizer='adam', learning_rate=0.001):
        """Compile the model"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Create optimizer
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print(f"✅ Model compiled with {optimizer} optimizer")
        return self.model
    
    def get_model_summary(self):
        """Get detailed model summary"""
        if self.model is None:
            return "Model not built yet."
        
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return '\n'.join(summary)

def create_model(num_classes, architecture='simple', input_shape=(28, 28, 1)):
    """Factory function to create and compile a model"""
    classifier = DoodleClassifier(num_classes, input_shape)
    model = classifier.build_model(architecture)
    model = classifier.compile_model()
    
    return model, classifier

def main():
    """Test model creation"""
    print("🎨 Testing Model Architecture")
    print("=" * 50)
    
    # Test different architectures
    architectures = ['simple', 'advanced', 'lightweight']
    num_classes = 10  # Test with 10 classes
    
    for arch in architectures:
        print(f"\n🧪 Testing {arch} architecture...")
        try:
            model, classifier = create_model(num_classes, arch)
            print(f"✅ {arch} model created successfully")
            print(f"   Parameters: {model.count_params():,}")
            
            # Test with dummy data
            import numpy as np
            dummy_input = np.random.random((1, 28, 28, 1))
            output = model.predict(dummy_input, verbose=0)
            print(f"   Output shape: {output.shape}")
            
        except Exception as e:
            print(f"❌ {arch} model failed: {e}")

if __name__ == "__main__":
    main()