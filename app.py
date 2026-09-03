from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

class StackingRegressor(BaseEstimator, RegressorMixin):
    """Stacking Regressor modeli"""
    
    def __init__(self, base_models, meta_model, cv=5, use_features_in_meta=True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv
        self.use_features_in_meta = use_features_in_meta
        self.trained_base_models = []
        self.is_fitted = False
    
    def fit(self, X, y):
        """Stacking modelini eğit"""
        
        print("Stacking modeli eğitiliyor...")
        
        # Initialize KFold
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # Create out-of-fold predictions
        oof_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        # Train base models with cross-validation
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"  Base model '{name}' eğitiliyor...")
            
            model_oof_predictions = np.zeros(X.shape[0])
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model on fold
                model.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                model_oof_predictions[val_idx] = model.predict(X_val_fold)
            
            oof_predictions[:, i] = model_oof_predictions
        
        # Train final base models on full dataset
        self.trained_base_models = []
        for name, model in self.base_models.items():
            print(f"  Final base model '{name}' eğitiliyor...")
            model.fit(X, y)
            self.trained_base_models.append((name, model))
        
        # Prepare meta-features
        if self.use_features_in_meta:
            # Combine out-of-fold predictions with original features
            meta_features = np.column_stack([oof_predictions, X.values])
        else:
            # Use only out-of-fold predictions
            meta_features = oof_predictions
        
        # Train meta-model
        print("  Meta-model eğitiliyor...")
        self.meta_model.fit(meta_features, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Stacking modeli ile tahmin yap"""
        
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Get predictions from all base models
        base_predictions = np.column_stack([
            model.predict(X) for _, model in self.trained_base_models
        ])
        
        # Prepare meta-features
        if self.use_features_in_meta:
            meta_features = np.column_stack([base_predictions, X.values])
        else:
            meta_features = base_predictions
        
        # Get final prediction from meta-model
        final_predictions = self.meta_model.predict(meta_features)
        
        return final_predictions

app = Flask(__name__)

# Global variables for model and preprocessing
model = None
scaler = None
label_encoder = None
feature_columns = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_and_preprocessing():
    """Model ve preprocessing nesnelerini yükle"""
    global model, scaler, label_encoder, feature_columns
    
    try:
        # Load the trained stacking model
        model = joblib.load(os.path.join(BASE_DIR, 'stacking_model.pkl'))
        
        # Load preprocessing objects
        scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
        label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))
        
        # Define feature columns
        feature_columns = [
            'NİTEL ARAŞTIMA\nDESENİ PUAN',
            'ARAŞTIRMANIN\nKAPSAMI',
            'ARAŞTIMACININ\nYETKİNLİĞİ',
            'BİLGİ GÜCÜ',
            'GÖRÜŞME\n SAYISI',
            'GÖRÜŞME\n SÜRESİ',
            'GÖZLEM\n SÜRESİ',
            'HOMOJENLİK/\nHETEROJENLİK',
            'KATILIMCI \nÖZGÜNLÜĞÜ',
            'VERİ\n ÇEŞİTLİLİĞİ',
            'VERİ KALİTESİ',
            'NİTEL ARAŞTIMA DESENİ'
        ]
        
        print("Model ve preprocessing nesneleri başarıyla yüklendi!")
        return True
        
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return False

# Load model on startup (required for Gunicorn and production)
load_model_and_preprocessing()

def prepare_input_data(form_data):
    """Form verilerini model için hazırla"""
    
    # Create input DataFrame with correct column names (matching the model training)
    input_data = {}
    
    # Research scope
    input_data['ARAŞTIRMANIN\nKAPSAMI'] = float(form_data.get('research_scope', 0))
    
    # Researcher competence
    input_data['ARAŞTIMACININ\nYETKİNLİĞİ'] = float(form_data.get('researcher_competence', 0))
    
    # Information power
    input_data['BİLGİ GÜCÜ'] = float(form_data.get('information_power', 0))
    
    # Interview count
    input_data['GÖRÜŞME\n SAYISI'] = float(form_data.get('interview_count', 0))
    
    # Interview duration
    input_data['GÖRÜŞME\n SÜRESİ'] = float(form_data.get('interview_duration', 0))
    
    # Observation duration
    input_data['GÖZLEM\n SÜRESİ'] = float(form_data.get('observation_duration', 0))
    
    # Homogeneity/Heterogeneity
    input_data['HOMOJENLİK/\nHETEROJENLİK'] = float(form_data.get('homogeneity', 0))
    
    # Participant uniqueness
    input_data['KATILIMCI \nÖZGÜNLÜĞÜ'] = float(form_data.get('participant_uniqueness', 0))
    
    # Data diversity
    input_data['VERİ\n ÇEŞİTLİLİĞİ'] = float(form_data.get('data_diversity', 0))
    
    # Data quality
    input_data['VERİ KALİTESİ'] = float(form_data.get('data_quality', 0))
    
    # Research design (categorical)
    research_design = form_data.get('research_design', '')
    if research_design in label_encoder.classes_:
        input_data['NİTEL ARAŞTIMA DESENİ'] = label_encoder.transform([research_design])[0]
    else:
        # Default to first class if not found
        input_data['NİTEL ARAŞTIMA DESENİ'] = 0
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    return df

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Tahmin yap"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Prepare input data
        input_df = prepare_input_data(form_data)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Round to nearest integer
        predicted_sample_size = int(round(prediction))
        
        # Ensure minimum sample size
        predicted_sample_size = max(1, predicted_sample_size)
        
        # Return result
        return jsonify({
            'success': True,
            'predicted_sample_size': predicted_sample_size,
            'confidence': 'Yüksek' if predicted_sample_size > 0 else 'Düşük'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/about')
def about():
    """Hakkında sayfası"""
    return render_template('about.html')

if __name__ == '__main__':
    # Load model and preprocessing
    if load_model_and_preprocessing():
        print("Web uygulaması başlatılıyor...")
        print("Uygulama http://localhost:8080 adresinde çalışacak")
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        print("Model yüklenemedi! Lütfen model dosyalarını kontrol edin.")
