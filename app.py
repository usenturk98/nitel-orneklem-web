from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
import threading
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

class StackingRegressor(BaseEstimator, RegressorMixin):
    """Stacking Regressor modeli"""
    
    def __init__(self, base_models=None, meta_model=None, cv=3, use_features_in_meta=True):
        self.base_models = base_models or {}
        self.meta_model = meta_model
        self.cv = cv
        self.use_features_in_meta = use_features_in_meta
        self.trained_base_models = []
        self.is_fitted = False
    
    def fit(self, X, y):
        """Stacking modelini eğit"""
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        oof_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            model_oof_predictions = np.zeros(X.shape[0])
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                model.fit(X_train_fold, y_train_fold)
                model_oof_predictions[val_idx] = model.predict(X_val_fold)
            oof_predictions[:, i] = model_oof_predictions
        
        self.trained_base_models = []
        for name, model in self.base_models.items():
            model.fit(X, y)
            self.trained_base_models.append((name, model))
        
        if self.use_features_in_meta:
            meta_features = np.column_stack([oof_predictions, X.values])
        else:
            meta_features = oof_predictions
        
        self.meta_model.fit(meta_features, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Stacking modeli ile tahmin yap"""
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmemiş!")
        
        base_predictions = np.column_stack([
            model.predict(X) for _, model in self.trained_base_models
        ])
        
        if self.use_features_in_meta:
            meta_features = np.column_stack([base_predictions, X.values])
        else:
            meta_features = base_predictions
        
        return self.meta_model.predict(meta_features)

# Register class in sys.modules so pickle can always deserialize it
setattr(sys.modules['__main__'], 'StackingRegressor', StackingRegressor)

app = Flask(__name__)

# Global variables for model and preprocessing
model = None
scaler = None
label_encoder = None
model_lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLUMNS = [
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

DESIGN_POINTS = {
    'Anlatı Araştırması': 7.5,
    'Etnografik Araştırma': 25.0,
    'Fenomenoloji': 22.5,
    'Gömülü Kuram': 25.0,
    'Örnek Olay': 17.5
}

def train_fallback_model():
    """Eğer pickle dosyaları yüklenemezse veri setinden modeli ultra hızlı eğitir"""
    global model, scaler, label_encoder
    print("Veri setinden model optimize şekilde eğitiliyor...")
    data_path = os.path.join(BASE_DIR, 'yeni_veri_seti_esit_orneklem.xlsx')
    if not os.path.exists(data_path):
        data_path = os.path.join(BASE_DIR, 'veri_seti.xlsx')
    
    df = pd.read_excel(data_path)
    df['NİTEL ARAŞTIMA\nDESENİ PUAN'] = pd.to_numeric(df['NİTEL ARAŞTIMA\nDESENİ PUAN'], errors='coerce').fillna(20.0)
    
    X = df[[
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
        'VERİ KALİTESİ'
    ]].copy()
    y = df['ÖRNEKLEM \nBÜYÜKLÜĞÜ'].copy()
    
    label_encoder = LabelEncoder()
    df['NİTEL ARAŞTIMA DESENİ_encoded'] = label_encoder.fit_transform(df['NİTEL ARAŞTIMA DESENİ'])
    X['NİTEL ARAŞTIMA DESENİ'] = df['NİTEL ARAŞTIMA DESENİ_encoded']
    
    base_models = {
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0)
    }
    
    trained_model = StackingRegressor(
        base_models=base_models,
        meta_model=GradientBoostingRegressor(n_estimators=30, random_state=42),
        cv=3,
        use_features_in_meta=True
    )
    trained_model.fit(X, y)
    
    scaler_obj = StandardScaler()
    scaler_obj.fit(X)
    
    model = trained_model
    scaler = scaler_obj
    
    # Save newly trained objects with current scikit-learn version
    try:
        joblib.dump(model, os.path.join(BASE_DIR, 'stacking_model.pkl'))
        joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
        joblib.dump(label_encoder, os.path.join(BASE_DIR, 'label_encoder.pkl'))
        print("Model başarıyla kaydedildi!")
    except Exception as e:
        print(f"Model kaydetme logu: {e}")
    
    return True

def ensure_model_loaded():
    """Modelin yüklü olduğunu garanti eder (Lazy Loading)"""
    global model, scaler, label_encoder
    if model is not None and label_encoder is not None:
        return True
        
    with model_lock:
        if model is not None and label_encoder is not None:
            return True
        try:
            model = joblib.load(os.path.join(BASE_DIR, 'stacking_model.pkl'))
            scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
            label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))
            print("Model dosyadan başarıyla yüklendi!")
            return True
        except Exception as e:
            print(f"Model dosyadan okunamadı ({e}), anında eğitiliyor...")
            return train_fallback_model()

# Modeli arka planda sessizce yükle / hazırla (Port açılışını asla engellemez)
try:
    threading.Thread(target=ensure_model_loaded, daemon=True).start()
except Exception:
    pass

def prepare_input_data(form_data):
    """Form verilerini model için hazırla"""
    input_data = {}
    
    research_design = form_data.get('research_design', 'Anlatı Araştırması')
    input_data['NİTEL ARAŞTIMA\nDESENİ PUAN'] = float(DESIGN_POINTS.get(research_design, 20.0))
    input_data['ARAŞTIRMANIN\nKAPSAMI'] = float(form_data.get('research_scope', 5))
    input_data['ARAŞTIMACININ\nYETKİNLİĞİ'] = float(form_data.get('researcher_competence', 5))
    input_data['BİLGİ GÜCÜ'] = float(form_data.get('information_power', 5))
    input_data['GÖRÜŞME\n SAYISI'] = float(form_data.get('interview_count', 10))
    input_data['GÖRÜŞME\n SÜRESİ'] = float(form_data.get('interview_duration', 45))
    input_data['GÖZLEM\n SÜRESİ'] = float(form_data.get('observation_duration', 0))
    input_data['HOMOJENLİK/\nHETEROJENLİK'] = float(form_data.get('homogeneity', 5))
    input_data['KATILIMCI \nÖZGÜNLÜĞÜ'] = float(form_data.get('participant_uniqueness', 5))
    input_data['VERİ\n ÇEŞİTLİLİĞİ'] = float(form_data.get('data_diversity', 5))
    input_data['VERİ KALİTESİ'] = float(form_data.get('data_quality', 5))
    
    if label_encoder is not None and hasattr(label_encoder, 'classes_') and research_design in label_encoder.classes_:
        input_data['NİTEL ARAŞTIMA DESENİ'] = label_encoder.transform([research_design])[0]
    else:
        # Sabit desen indeksleri
        design_map = {
            'Anlatı Araştırması': 0,
            'Etnografik Araştırma': 1,
            'Fenomenoloji': 2,
            'Gömülü Kuram': 3,
            'Örnek Olay': 4
        }
        input_data['NİTEL ARAŞTIMA DESENİ'] = design_map.get(research_design, 0)
    
    df = pd.DataFrame([input_data])
    return df[FEATURE_COLUMNS]

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Sağlık kontrolü"""
    return jsonify({'status': 'ok', 'model_ready': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Tahmin yap"""
    global model
    try:
        ensure_model_loaded()
            
        form_data = request.form.to_dict()
        input_df = prepare_input_data(form_data)
        
        prediction = model.predict(input_df)[0]
        predicted_sample_size = int(round(prediction))
        predicted_sample_size = max(1, predicted_sample_size)
        
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
    port = int(os.environ.get('PORT', 8080))
    print(f"Web uygulaması {port} portunda başlatılıyor...")
    app.run(debug=True, host='0.0.0.0', port=port)
