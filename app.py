import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, send_file, request, jsonify
import plotly.express as px
import plotly.graph_objects as go
import plotly
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import base64
from io import BytesIO
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

app = Flask(__name__)

print("\n" + "=" * 70)
print("🚦 DASHBOARD ANALYSE ET PRÉDICTION DE TRAFIC - VERSION COMPLÈTE")
print("=" * 70)


# ============================================
# 1. CHERCHER ET CHARGER LE FICHIER
# ============================================

def find_csv_file():
    """Cherche le fichier CSV dans uploads/ ou le dossier courant"""
    print("🔍 Recherche du fichier CSV...")

    uploads_dir = "uploads"
    if os.path.exists(uploads_dir) and os.path.isdir(uploads_dir):
        csv_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith('.csv')]

        if csv_files:
            print(f"✅ {len(csv_files)} fichier(s) CSV trouvé(s) dans uploads/")
            for csv_file in csv_files:
                file_path = os.path.join(uploads_dir, csv_file)
                size_kb = os.path.getsize(file_path) / 1024
                print(f"   • {csv_file} ({size_kb:.1f} KB)")

            selected = csv_files[0]
            file_path = os.path.join(uploads_dir, selected)
            print(f"📄 Sélection: {file_path}")
            return file_path

    csv_files = [f for f in os.listdir('.') if f.lower().endswith('.csv')]

    if csv_files:
        print(f"✅ {len(csv_files)} fichier(s) CSV trouvé(s) dans le dossier courant")
        selected = csv_files[0]
        print(f"📄 Sélection: {selected}")
        return selected

    print("❌ Aucun fichier CSV trouvé")
    return None


def load_csv_file(filepath):
    """Charge un fichier CSV"""
    print(f"📥 Chargement de: {filepath}")

    if not os.path.exists(filepath):
        print(f"❌ Fichier non trouvé: {filepath}")
        return None

    file_size = os.path.getsize(filepath)
    print(f"   Taille: {file_size / 1024:.1f} KB")

    try:
        # Essayer de lire avec différentes options
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')

        if len(df.columns) > 0 and len(df) > 0:
            print(f"✅ Chargement réussi")
            print(f"   Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
            print(f"   Colonnes: {df.columns.tolist()}")
            return df
        else:
            print("❌ Fichier vide ou invalide")
            return None

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None


def create_demo_dataset():
    """Crée un dataset de démonstration"""
    print("📊 Création d'un dataset de démonstration...")

    dates = pd.date_range(
        start='2024-01-01 00:00:00',
        end='2024-03-31 23:00:00',
        freq='H'
    )

    n = len(dates)

    df = pd.DataFrame({
        'date_time': dates,
        'traffic_volume': np.clip(
            1200 + 800 * np.sin(2 * np.pi * (dates.hour - 7) / 12) +
            600 * np.sin(2 * np.pi * (dates.hour - 17) / 12) +
            300 * (1 - 0.4 * (dates.dayofweek >= 5)) +
            np.random.normal(0, 200, n),
            200, 3200
        ).astype(int),
        'speed': np.random.randint(30, 100, n),
        'temp': np.random.randint(-5, 35, n),
        'rain_1h': np.random.exponential(0.5, n).round(1),
        'snow_1h': np.where(dates.month.isin([1, 2, 12]),
                            np.random.exponential(0.1, n), 0).round(1),
        'clouds_all': np.random.randint(0, 100, n),
        'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain', 'Snow', 'Mist'], n)
    })

    print(f"✅ Dataset créé: {len(df)} lignes")
    return df, "Dataset généré"


# Trouver et charger le fichier
csv_path = find_csv_file()

if csv_path:
    TRAFFIC_DATA = load_csv_file(csv_path)
    if TRAFFIC_DATA is not None:
        DATA_SOURCE = csv_path
    else:
        TRAFFIC_DATA, DATA_SOURCE = create_demo_dataset()
else:
    TRAFFIC_DATA, DATA_SOURCE = create_demo_dataset()

print(f"\n📊 DONNÉES DISPONIBLES:")
print(f"   Source: {DATA_SOURCE}")
print(f"   Enregistrements: {len(TRAFFIC_DATA)}")
print("=" * 70)


# ============================================
# 2. PRÉPARATION DES DONNÉES (COMPLÈTE)
# ============================================

def prepare_data(df):
    """Prépare les données pour l'analyse - VERSION COMPLÈTE"""
    df = df.copy()

    print("\n🔧 Préparation des données...")

    # 1. Convertir la colonne date_time en datetime
    if 'date_time' in df.columns:
        print(f"📅 Conversion de la colonne 'date_time'...")

        # Essayer différents formats de date
        original_dtype = str(df['date_time'].dtype)
        print(f"   Type original: {original_dtype}")
        print(f"   Premier échantillon: {df['date_time'].iloc[0]}")

        try:
            # Essayer de convertir en datetime avec dayfirst=True
            df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True, errors='coerce')

            # Vérifier combien de conversions ont réussi
            valid_dates = df['date_time'].notna().sum()
            total_dates = len(df)
            success_rate = (valid_dates / total_dates) * 100

            print(f"   ✅ Conversion réussie: {success_rate:.1f}% ({valid_dates}/{total_dates})")
            print(f"   Première date valide: {df['date_time'].min()}")
            print(f"   Dernière date valide: {df['date_time'].max()}")

            # Si moins de 50% des dates sont valides, créer une nouvelle timeline
            if success_rate < 50:
                print(f"   ⚠️  Taux de conversion faible, création d'une timeline...")
                df['date_time'] = pd.date_range(start='2012-10-02', periods=len(df), freq='H')

        except Exception as e:
            print(f"   ❌ Erreur de conversion: {e}")
            print(f"   ➕ Création d'une timeline...")
            df['date_time'] = pd.date_range(start='2012-10-02', periods=len(df), freq='H')
    else:
        print(f"   ⚠️  Colonne 'date_time' non trouvée")
        print(f"   ➕ Création d'une timeline...")
        df['date_time'] = pd.date_range(start='2012-10-02', periods=len(df), freq='H')

    # 2. Vérifier et renommer les colonnes de trafic
    if 'traffic_volume' in df.columns:
        print(f"   🚗 Colonne 'traffic_volume' trouvée")
    else:
        # Chercher d'autres noms possibles
        traffic_cols = [col for col in df.columns if 'traffic' in col.lower() or 'volume' in col.lower()]
        if traffic_cols:
            df['traffic_volume'] = df[traffic_cols[0]]
            print(f"   ✅ Colonne '{traffic_cols[0]}' utilisée comme traffic_volume")
        else:
            # Prendre la première colonne numérique (hors date)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'date_time']

            if numeric_cols:
                df['traffic_volume'] = df[numeric_cols[0]]
                print(f"   ⚠️  Colonne '{numeric_cols[0]}' utilisée comme traffic_volume")
            else:
                # Créer des données factices
                df['traffic_volume'] = np.random.randint(100, 5000, len(df))
                print(f"   ➕ Données de trafic générées")

    # 3. Vérifier la colonne speed
    if 'speed' not in df.columns:
        # Chercher d'autres noms
        speed_cols = [col for col in df.columns if 'speed' in col.lower()]
        if speed_cols:
            df['speed'] = df[speed_cols[0]]
        else:
            # Générer une vitesse réaliste basée sur le trafic
            if 'traffic_volume' in df.columns:
                max_traffic = df['traffic_volume'].max()
                df['speed'] = 80 - (df['traffic_volume'] / max_traffic * 40)
            else:
                df['speed'] = 60

    # 4. Vérifier la colonne temp
    if 'temp' not in df.columns:
        # Chercher d'autres noms
        temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
        if temp_cols:
            df['temp'] = df[temp_cols[0]]
        else:
            # Générer des températures réalistes basées sur le mois
            if 'date_time' in df.columns:
                df['temp'] = np.where(df['date_time'].dt.month.isin([12, 1, 2]),
                                      np.random.randint(-5, 10, len(df)),
                                      np.random.randint(10, 35, len(df)))
            else:
                df['temp'] = np.random.randint(-5, 35, len(df))

    # 5. Extraire les caractéristiques temporelles
    print(f"   ⏰ Extraction des composantes temporelles...")

    try:
        df["hour"] = df["date_time"].dt.hour
        df["month"] = df["date_time"].dt.month
        df["day_of_week"] = df["date_time"].dt.dayofweek
        df["day_of_year"] = df["date_time"].dt.dayofyear

        # Noms des jours
        df["day_name"] = df["day_of_week"].map({
            0: 'Lundi', 1: 'Mardi', 2: 'Mercredi',
            3: 'Jeudi', 4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'
        })

        # Week-end
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Catégorie d'heure
        df["hour_category"] = pd.cut(
            df["hour"],
            bins=[0, 6, 9, 16, 19, 24],
            labels=['Nuit', 'Matin', 'Journée', 'Soir', 'Nuit tard'],
            right=False
        )

        print(f"   ✅ Extraction temporelle réussie")

    except Exception as e:
        print(f"   ❌ Erreur lors de l'extraction temporelle: {e}")
        print(f"   ➕ Création de valeurs temporelles factices...")

        # Créer des valeurs factices
        df["hour"] = np.random.randint(0, 24, len(df))
        df["month"] = np.random.randint(1, 13, len(df))
        df["day_of_week"] = np.random.randint(0, 7, len(df))
        df["day_name"] = df["day_of_week"].map({
            0: 'Lundi', 1: 'Mardi', 2: 'Mercredi',
            3: 'Jeudi', 4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'
        })
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 6. Vérifier les autres colonnes météo
    for col in ['rain_1h', 'snow_1h', 'clouds_all', 'weather_main']:
        if col not in df.columns:
            if col == 'rain_1h':
                df[col] = np.random.exponential(0.5, len(df)).round(1)
            elif col == 'snow_1h':
                df[col] = np.where(df['month'].isin([1, 2, 12]) if 'month' in df.columns else False,
                                   np.random.exponential(0.1, len(df)), 0).round(1)
            elif col == 'clouds_all':
                df[col] = np.random.randint(0, 100, len(df))
            elif col == 'weather_main':
                df[col] = np.random.choice(['Clear', 'Clouds', 'Rain', 'Snow', 'Mist'], len(df))

    # 7. Nettoyer les données
    initial_count = len(df)
    df = df.dropna(subset=['date_time', 'traffic_volume'])
    df = df.sort_values('date_time').reset_index(drop=True)

    removed = initial_count - len(df)
    if removed > 0:
        print(f"   🗑️  {removed} lignes avec valeurs manquantes supprimées")

    print(f"   ✅ Données prêtes: {len(df)} lignes")
    print(f"   📅 Période: {df['date_time'].min()} - {df['date_time'].max()}")
    print(f"   🚗 Trafic moyen: {df['traffic_volume'].mean():.0f}")
    print(f"   🏎️  Vitesse moyenne: {df['speed'].mean():.1f}" if 'speed' in df.columns else "")
    print(f"   🌡️  Température moyenne: {df['temp'].mean():.1f}°C" if 'temp' in df.columns else "")

    return df


# Préparer les données
PREPARED_DATA = prepare_data(TRAFFIC_DATA)

print(f"\n✅ PRÉPARATION TERMINÉE")
print(f"   Source: {DATA_SOURCE}")
print(f"   Enregistrements: {len(PREPARED_DATA):,}")
print(f"   Colonnes disponibles: {PREPARED_DATA.columns.tolist()}")
print("=" * 70)


# ============================================
# 3. MODÈLES POUR SIMULATION (TRAFIC + VITESSE)
# ============================================

class TrafficSimulator:
    """Simulateur pour prédire trafic et vitesse sur 24h"""

    def __init__(self, df):
        self.df = df.copy()
        self.models = {}
        self.scalers = {}
        self.seq_length = 24

    def prepare_simulation_data(self):
        """Prépare les données pour la simulation multi-cibles"""
        print("\n🤖 Préparation des données pour la simulation...")

        # Sélectionner les features
        features = ['traffic_volume', 'speed', 'hour', 'day_of_week', 'month']
        if 'temp' in self.df.columns:
            features.append('temp')
        if 'rain_1h' in self.df.columns:
            features.append('rain_1h')
        if 'snow_1h' in self.df.columns:
            features.append('snow_1h')

        # Garder seulement les colonnes disponibles
        available_features = [f for f in features if f in self.df.columns]

        print(f"   Features disponibles: {available_features}")

        # Créer le dataset pour la simulation
        data = self.df[available_features].copy()

        # Normalisation
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Créer des séquences
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:i + self.seq_length])
            y.append(scaled_data[i + self.seq_length][:2])  # traffic_volume et speed

        X = np.array(X)
        y = np.array(y)

        print(f"   Forme des données: X={X.shape}, y={y.shape}")

        return X, y, scaler, available_features

    def build_multi_output_model(self, input_shape):
        """Construit un modèle pour prédire trafic ET vitesse"""
        print("   🏗️  Construction du modèle multi-sorties...")

        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(2)  # Deux sorties: traffic_volume et speed
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train_models(self):
        """Entraîne les modèles de simulation"""
        try:
            # Préparer les données
            X, y, scaler, features = self.prepare_simulation_data()

            # Sauvegarder le scaler
            self.scalers['multi'] = scaler
            self.features = features

            # Division train/test
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Construire et entraîner le modèle
            model = self.build_multi_output_model((self.seq_length, X_train.shape[2]))

            print("   🏋️  Entraînement du modèle multi-sorties...")

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=20,
                batch_size=32,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ModelCheckpoint('best_simulation_model.h5', save_best_only=True)
                ],
                verbose=0
            )

            # Évaluation
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            print(f"   📊 Performance du modèle: Loss={test_loss:.4f}, MAE={test_mae:.4f}")

            self.models['multi'] = model
            self.history = history.history

            # Sauvegarder aussi un modèle simple pour le trafic seul (backup)
            self._train_simple_model()

            return True

        except Exception as e:
            print(f"   ❌ Erreur lors de l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _train_simple_model(self):
        """Entraîne un modèle simple pour le trafic seulement (backup)"""
        try:
            # Features simplifiées pour le trafic
            features = ['traffic_volume', 'hour', 'day_of_week', 'month']
            data = self.df[features].copy()

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            X, y = [], []
            for i in range(len(scaled_data) - self.seq_length):
                X.append(scaled_data[i:i + self.seq_length])
                y.append(scaled_data[i + self.seq_length][0])  # traffic_volume seulement

            X = np.array(X)
            y = np.array(y)

            # Modèle simple
            model = Sequential([
                LSTM(50, input_shape=(self.seq_length, X.shape[2])),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Entraînement rapide
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            self.models['traffic_only'] = model
            self.scalers['traffic_only'] = scaler

        except Exception as e:
            print(f"   ⚠️  Erreur modèle simple: {e}")

    def predict_next_24h(self, start_date=None):
        """Prédit le trafic et la vitesse pour les 24 prochaines heures"""
        print("   🔮 Prédiction des 24 prochaines heures...")

        try:
            if 'multi' not in self.models:
                print("   ⚠️  Modèle multi-sorties non disponible, utilisation du modèle simple...")
                return self._predict_simple_24h()

            # Utiliser les dernières 24 heures comme point de départ
            last_data = self.df[self.features].tail(self.seq_length).values

            # Normaliser
            scaled_data = self.scalers['multi'].transform(last_data)

            # Préparer l'input
            X_input = scaled_data.reshape(1, self.seq_length, -1)

            # Prédictions pour les 24 prochaines heures
            predictions = []
            current_sequence = X_input.copy()

            for hour in range(24):
                # Prédire l'heure suivante
                pred = self.models['multi'].predict(current_sequence, verbose=0)
                predictions.append(pred[0])

                # Mettre à jour la séquence
                new_row = np.zeros((1, 1, len(self.features)))
                new_row[0, 0, :2] = pred[0]  # traffic_volume et speed prédits

                # Pour les autres features, utiliser des valeurs typiques ou propagées
                if len(self.features) > 2:
                    # Pour l'heure, incrémenter
                    hour_idx = self.features.index('hour') if 'hour' in self.features else -1
                    if hour_idx != -1:
                        current_hour = (current_sequence[0, -1, hour_idx] * 23)  # dénormaliser
                        next_hour = (current_hour + 1) % 24
                        new_row[0, 0, hour_idx] = next_hour / 23  # renormaliser

                    # Pour jour de la semaine
                    dow_idx = self.features.index('day_of_week') if 'day_of_week' in self.features else -1
                    if dow_idx != -1:
                        # Simuler le passage du temps
                        current_dow = current_sequence[0, -1, dow_idx] * 6  # dénormaliser
                        # Pour une prédiction de 24h, on pourrait incrémenter le jour si on dépasse minuit
                        # Pour simplifier, on garde le même jour
                        new_row[0, 0, dow_idx] = current_dow / 6  # renormaliser

                # Ajouter la nouvelle prédiction à la séquence
                current_sequence = np.concatenate([
                    current_sequence[:, 1:, :],  # Supprimer la première heure
                    new_row  # Ajouter la nouvelle prédiction
                ], axis=1)

            # Convertir les prédictions en array
            predictions = np.array(predictions)

            # Dénormaliser
            dummy_array = np.zeros((len(predictions), len(self.features)))
            dummy_array[:, :2] = predictions  # traffic_volume et speed
            denorm_predictions = self.scalers['multi'].inverse_transform(dummy_array)

            # Extraire trafic et vitesse
            traffic_predictions = denorm_predictions[:, 0]
            speed_predictions = denorm_predictions[:, 1]

            # Générer les dates/heures
            if start_date is None:
                last_date = self.df['date_time'].iloc[-1]
                start_date = last_date + pd.Timedelta(hours=1)

            future_dates = pd.date_range(
                start=start_date,
                periods=24,
                freq='H'
            )

            # Catégoriser l'état du trafic
            traffic_categories = []
            speed_categories = []

            for i in range(24):
                # Catégorisation trafic
                traffic = traffic_predictions[i]
                if traffic < 1000:
                    traffic_cat = "Fluide 🟢"
                elif traffic < 2000:
                    traffic_cat = "Modéré 🟡"
                elif traffic < 3000:
                    traffic_cat = "Dense 🟠"
                else:
                    traffic_cat = "Saturé 🔴"
                traffic_categories.append(traffic_cat)

                # Catégorisation vitesse
                speed = speed_predictions[i]
                if speed > 70:
                    speed_cat = "Rapide 🟢"
                elif speed > 50:
                    speed_cat = "Normale 🟡"
                elif speed > 30:
                    speed_cat = "Lente 🟠"
                else:
                    speed_cat = "Très lente 🔴"
                speed_categories.append(speed_cat)

            # Créer le DataFrame de résultats
            results_df = pd.DataFrame({
                'date_time': future_dates,
                'hour': future_dates.hour,
                'traffic_predicted': traffic_predictions.round(0),
                'speed_predicted': speed_predictions.round(1),
                'traffic_category': traffic_categories,
                'speed_category': speed_categories,
                'day_name': future_dates.strftime('%A')
            })

            # Traduire les jours
            day_translation = {
                'Monday': 'Lundi', 'Tuesday': 'Mardi', 'Wednesday': 'Mercredi',
                'Thursday': 'Jeudi', 'Friday': 'Vendredi', 'Saturday': 'Samedi',
                'Sunday': 'Dimanche'
            }
            results_df['day_name'] = results_df['day_name'].map(day_translation)

            print("   ✅ Prédictions générées avec succès")
            return results_df

        except Exception as e:
            print(f"   ❌ Erreur lors de la prédiction: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _predict_simple_24h(self):
        """Prédiction simple avec le modèle de trafic seulement"""
        try:
            # Features pour le modèle simple
            features = ['traffic_volume', 'hour', 'day_of_week', 'month']
            last_data = self.df[features].tail(self.seq_length).values

            scaler = self.scalers['traffic_only']
            scaled_data = scaler.transform(last_data)

            X_input = scaled_data.reshape(1, self.seq_length, -1)
            model = self.models['traffic_only']

            # Prédictions
            traffic_predictions = []
            current_sequence = X_input.copy()

            for hour in range(24):
                pred = model.predict(current_sequence, verbose=0)[0, 0]
                traffic_predictions.append(pred)

                # Mettre à jour la séquence
                new_row = np.zeros((1, 1, len(features)))
                new_row[0, 0, 0] = pred  # traffic_volume

                # Mettre à jour l'heure
                current_hour = (current_sequence[0, -1, 1] * 23)
                next_hour = (current_hour + 1) % 24
                new_row[0, 0, 1] = next_hour / 23

                # Autres features (garder constantes)
                for i in range(2, len(features)):
                    new_row[0, 0, i] = current_sequence[0, -1, i]

                current_sequence = np.concatenate([
                    current_sequence[:, 1:, :],
                    new_row
                ], axis=1)

            # Dénormaliser
            traffic_predictions = np.array(traffic_predictions).reshape(-1, 1)
            dummy_array = np.zeros((len(traffic_predictions), len(features)))
            dummy_array[:, 0] = traffic_predictions[:, 0]
            denorm_predictions = scaler.inverse_transform(dummy_array)
            traffic_predictions = denorm_predictions[:, 0]

            # Estimer la vitesse basée sur le trafic
            avg_speed = self.df['speed'].mean() if 'speed' in self.df.columns else 60
            max_traffic = self.df['traffic_volume'].max()
            speed_predictions = avg_speed - (traffic_predictions / max_traffic * 40)

            # Générer les dates
            last_date = self.df['date_time'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=24, freq='H')

            # Catégoriser
            traffic_categories = []
            speed_categories = []

            for i in range(24):
                traffic = traffic_predictions[i]
                if traffic < 1000:
                    traffic_cat = "Fluide 🟢"
                elif traffic < 2000:
                    traffic_cat = "Modéré 🟡"
                elif traffic < 3000:
                    traffic_cat = "Dense 🟠"
                else:
                    traffic_cat = "Saturé 🔴"
                traffic_categories.append(traffic_cat)

                speed = speed_predictions[i]
                if speed > 70:
                    speed_cat = "Rapide 🟢"
                elif speed > 50:
                    speed_cat = "Normale 🟡"
                elif speed > 30:
                    speed_cat = "Lente 🟠"
                else:
                    speed_cat = "Très lente 🔴"
                speed_categories.append(speed_cat)

            results_df = pd.DataFrame({
                'date_time': future_dates,
                'hour': future_dates.hour,
                'traffic_predicted': traffic_predictions.round(0),
                'speed_predicted': speed_predictions.round(1),
                'traffic_category': traffic_categories,
                'speed_category': speed_categories,
                'day_name': future_dates.strftime('%A')
            })

            return results_df

        except Exception as e:
            print(f"   ❌ Erreur prédiction simple: {e}")
            return None

    def generate_simulation_graphs(self, predictions_df):
        """Génère les graphiques pour la simulation"""
        print("   📈 Génération des graphiques de simulation...")

        graphs = {}

        try:
            # 1. Graphique combiné trafic + vitesse
            plt.figure(figsize=(14, 8))

            # Trafic
            plt.subplot(2, 1, 1)
            bars = plt.bar(range(24), predictions_df['traffic_predicted'],
                           color=[
                               '#4CAF50' if t < 1000 else '#FFC107' if t < 2000 else '#FF9800' if t < 3000 else '#F44336'
                               for t in predictions_df['traffic_predicted']])
            plt.title('Prédiction du Trafic sur 24h', fontsize=14, fontweight='bold')
            plt.xlabel('Heure')
            plt.ylabel('Trafic (véhicules/h)')
            plt.xticks(range(24), predictions_df['hour'], rotation=45)
            plt.grid(True, alpha=0.3, axis='y')

            # Ajouter les valeurs sur les barres
            for bar, val in zip(bars, predictions_df['traffic_predicted']):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                         f'{int(val)}', ha='center', va='bottom', fontsize=9)

            # Vitesse
            plt.subplot(2, 1, 2)
            plt.plot(range(24), predictions_df['speed_predicted'], 'b-o', linewidth=2, markersize=6)
            plt.title('Prédiction de la Vitesse sur 24h', fontsize=14, fontweight='bold')
            plt.xlabel('Heure')
            plt.ylabel('Vitesse (km/h)')
            plt.xticks(range(24), predictions_df['hour'], rotation=45)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, max(predictions_df['speed_predicted']) * 1.2)

            # Ajouter les valeurs
            for i, (hour, speed) in enumerate(zip(predictions_df['hour'], predictions_df['speed_predicted'])):
                plt.text(i, speed + 2, f'{speed:.0f} km/h', ha='center', fontsize=9)

            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            graphs['combined_prediction'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            # 2. Graphique de synthèse avec état du trafic
            fig, ax = plt.subplots(figsize=(14, 6))

            # Créer une carte de chaleur simplifiée
            hours = predictions_df['hour'].tolist()
            traffic_status = []
            for cat in predictions_df['traffic_category']:
                if '🟢' in cat:
                    traffic_status.append(0)  # Fluide
                elif '🟡' in cat:
                    traffic_status.append(1)  # Modéré
                elif '🟠' in cat:
                    traffic_status.append(2)  # Dense
                else:
                    traffic_status.append(3)  # Saturé

            # Créer un graphique à barres colorées
            colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
            bars = ax.bar(hours, [1] * 24, color=[colors[s] for s in traffic_status])

            # Ajouter les étiquettes
            for i, (hour, status, cat) in enumerate(zip(hours, traffic_status, predictions_df['traffic_category'])):
                ax.text(hour, 0.5, cat.replace(' 🟢', '').replace(' 🟡', '').replace(' 🟠', '').replace(' 🔴', ''),
                        ha='center', va='center', color='white', fontweight='bold', fontsize=10)
                ax.text(hour, 0.8, f"{predictions_df['traffic_predicted'].iloc[i]:.0f}",
                        ha='center', va='center', color='white', fontsize=9)

            ax.set_title('État du Trafic Prédit par Heure', fontsize=14, fontweight='bold')
            ax.set_xlabel('Heure')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_xticks(hours)
            ax.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45)

            # Légende
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, color='#4CAF50', label='Fluide (<1000)'),
                plt.Rectangle((0, 0), 1, 1, color='#FFC107', label='Modéré (1000-2000)'),
                plt.Rectangle((0, 0), 1, 1, color='#FF9800', label='Dense (2000-3000)'),
                plt.Rectangle((0, 0), 1, 1, color='#F44336', label='Saturé (>3000)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            graphs['traffic_status'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            # 3. Graphique circulaire récapitulatif
            plt.figure(figsize=(10, 8))

            # Compter les catégories
            traffic_counts = predictions_df['traffic_category'].value_counts()

            # Extraire juste les noms sans émojis pour les labels
            labels = []
            for cat in traffic_counts.index:
                labels.append(cat.replace(' 🟢', '').replace(' 🟡', '').replace(' 🟠', '').replace(' 🔴', ''))

            colors = []
            for cat in traffic_counts.index:
                if '🟢' in cat:
                    colors.append('#4CAF50')
                elif '🟡' in cat:
                    colors.append('#FFC107')
                elif '🟠' in cat:
                    colors.append('#FF9800')
                else:
                    colors.append('#F44336')

            plt.pie(traffic_counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 12})
            plt.title('Répartition des États de Trafic sur 24h', fontsize=14, fontweight='bold')
            plt.axis('equal')

            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            graphs['traffic_pie'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            print(f"   ✅ {len(graphs)} graphiques de simulation générés")
            return graphs

        except Exception as e:
            print(f"   ❌ Erreur génération graphiques: {e}")
            import traceback
            traceback.print_exc()
            return {}


# Initialiser le simulateur
print("\n🚀 Initialisation du simulateur de trafic...")
SIMULATOR = TrafficSimulator(PREPARED_DATA)


# ============================================
# 4. MODÈLE CNN-LSTM AMÉLIORÉ (SANS ERREUR)
# ============================================

def prepare_prediction_data_safe(df):
    """Prépare les données pour la prédiction - VERSION SÉCURISÉE"""
    print("\n🤖 Préparation des données pour la prédiction...")

    # Faire une copie
    df_pred = df.copy()

    print("   🔧 Feature engineering...")

    # 1. Vérifier les colonnes essentielles
    if 'traffic_volume' not in df_pred.columns:
        print("   ❌ ERREUR: Colonne 'traffic_volume' manquante")
        raise ValueError("Colonne 'traffic_volume' requise")

    # 2. Créer les features temporelles
    if 'hour' not in df_pred.columns:
        if 'date_time' in df_pred.columns:
            try:
                df_pred['date_time'] = pd.to_datetime(df_pred['date_time'], errors='coerce')
                df_pred['hour'] = df_pred['date_time'].dt.hour
                df_pred['day_of_week'] = df_pred['date_time'].dt.dayofweek
                df_pred['month'] = df_pred['date_time'].dt.month
                print("   ✅ Features temporelles extraites")
            except:
                df_pred['hour'] = np.random.randint(0, 24, len(df_pred))
                df_pred['day_of_week'] = np.random.randint(0, 7, len(df_pred))
                df_pred['month'] = np.random.randint(1, 13, len(df_pred))
                print("   ⚠️  Features temporelles générées")
        else:
            df_pred['hour'] = np.random.randint(0, 24, len(df_pred))
            df_pred['day_of_week'] = np.random.randint(0, 7, len(df_pred))
            df_pred['month'] = np.random.randint(1, 13, len(df_pred))
            print("   ⚠️  Features temporelles générées (pas de date_time)")

    # 3. Sélectionner les features disponibles
    potential_features = ['traffic_volume', 'speed', 'temp', 'hour', 'day_of_week', 'month']
    available_features = []

    for feat in potential_features:
        if feat in df_pred.columns:
            available_features.append(feat)
        else:
            print(f"   ⚠️  Feature '{feat}' non disponible")

    # S'assurer qu'on a au moins 3 features
    if len(available_features) < 3:
        # Ajouter des features supplémentaires
        numeric_cols = df_pred.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in available_features and col not in ['date_time']:
                available_features.append(col)
                print(f"   ➕ Ajout de '{col}' comme feature")

    print(f"   ✅ Features finales ({len(available_features)}): {available_features}")

    # 4. Sélectionner et nettoyer les données
    try:
        df_selected = df_pred[available_features].copy()
    except KeyError as e:
        print(f"   ❌ Erreur: {e}")
        print(f"   Colonnes disponibles: {df_pred.columns.tolist()}")
        raise

    # Remplacer les NaN
    df_selected = df_selected.fillna(method='ffill').fillna(method='bfill').fillna(0)

    if len(df_selected) < 100:
        raise ValueError(f"Pas assez de données ({len(df_selected)} < 100)")

    print(f"   📈 Données finales: {len(df_selected)} échantillons")

    return {
        'data': df_selected,
        'features': available_features
    }


def create_sequences_safe(data, seq_len=24):
    """Crée des séquences pour l'apprentissage - VERSION SÉCURISÉE"""
    print(f"   🔄 Création des séquences (longueur={seq_len})...")

    # Convertir en numpy array
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data

    X, y = [], []
    for i in range(len(data_array) - seq_len):
        X.append(data_array[i:i + seq_len])
        y.append(data_array[i + seq_len][0])  # Première colonne = traffic_volume

    print(f"   ✅ {len(X)} séquences créées")
    return np.array(X), np.array(y)


def build_cnn_lstm_model_safe(input_shape):
    """Construit le modèle CNN-LSTM - VERSION SÉCURISÉE"""
    print("   🏗️  Construction du modèle...")

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    return model


def train_prediction_model_safe(df, seq_length=24, test_size=0.2):
    """Entraîne le modèle CNN-LSTM - VERSION SÉCURISÉE"""
    print("\n🧠 Entraînement du modèle CNN-LSTM...")

    try:
        # 1. Préparer les données
        prep_data = prepare_prediction_data_safe(df)
        df_selected = prep_data['data']
        features = prep_data['features']

        print(f"   🔍 Données préparées:")
        print(f"      Shape: {df_selected.shape}")
        print(f"      Features: {features}")

        # 2. Normalisation
        print("   📏 Normalisation des données...")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_selected)

        # 3. Créer les séquences
        X, y = create_sequences_safe(scaled_data, seq_length)

        print(f"   📊 Séquences créées:")
        print(f"      X shape: {X.shape}")
        print(f"      y shape: {y.shape}")

        # 4. Division train/test
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"   📈 Division des données:")
        print(f"      Train: {len(X_train)} séquences")
        print(f"      Test: {len(X_test)} séquences")

        # 5. Construire et entraîner le modèle
        model = build_cnn_lstm_model_safe((seq_length, X_train.shape[2]))

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('best_cnn_lstm_model.h5', monitor='val_loss', save_best_only=True)
        ]

        print("   🏋️  Entraînement du modèle...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=15,  # Réduit pour aller plus vite
            batch_size=32,
            callbacks=callbacks,
            verbose=0  # Silent mode
        )

        print(f"   ✅ Modèle entraîné ({len(history.history['loss'])} epochs)")

        # 6. Prédictions
        print("   🔮 Prédictions...")
        y_train_pred = model.predict(X_train, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)

        # 7. Inverse scaling
        # Pour train
        scaled_full_train = np.zeros((len(y_train_pred), df_selected.shape[1]))
        scaled_full_train[:, 0] = y_train_pred[:, 0]
        y_train_pred_real = scaler.inverse_transform(scaled_full_train)[:, 0]

        scaled_actual_train = np.zeros((len(y_train), df_selected.shape[1]))
        scaled_actual_train[:, 0] = y_train
        y_train_real = scaler.inverse_transform(scaled_actual_train)[:, 0]

        # Pour test
        scaled_full_test = np.zeros((len(y_test_pred), df_selected.shape[1]))
        scaled_full_test[:, 0] = y_test_pred[:, 0]
        y_test_pred_real = scaler.inverse_transform(scaled_full_test)[:, 0]

        scaled_actual_test = np.zeros((len(y_test), df_selected.shape[1]))
        scaled_actual_test[:, 0] = y_test
        y_test_real = scaler.inverse_transform(scaled_actual_test)[:, 0]

        # 8. Calculer les métriques
        train_rmse = np.sqrt(np.mean((y_train_real - y_train_pred_real) ** 2))
        train_mae = np.mean(np.abs(y_train_real - y_train_pred_real))

        test_rmse = np.sqrt(np.mean((y_test_real - y_test_pred_real) ** 2))
        test_mae = np.mean(np.abs(y_test_real - y_test_pred_real))

        print(f"   📊 Métriques d'entraînement:")
        print(f"      RMSE: {train_rmse:.2f}")
        print(f"      MAE: {train_mae:.2f}")

        print(f"   📊 Métriques de test:")
        print(f"      RMSE: {test_rmse:.2f}")
        print(f"      MAE: {test_mae:.2f}")

        # 9. Dates pour l'affichage
        if 'date_time' in df.columns:
            dates = df['date_time'].values[seq_length:]
            train_dates = dates[:split]
            test_dates = dates[split:]
        else:
            train_dates = np.arange(len(y_train_real))
            test_dates = np.arange(len(y_test_real)) + len(y_train_real)

        # 10. Préparer les résultats
        results = {
            'model': model,
            'history': history.history,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'y_train_actual': y_train_real,
            'y_train_pred': y_train_pred_real,
            'y_test_actual': y_test_real,
            'y_test_pred': y_test_pred_real,
            'metrics': {
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'test_rmse': test_rmse,
                'test_mae': test_mae
            },
            'scaler': scaler,
            'features': features,
            'seq_length': seq_length
        }

        # 11. Sauvegarder les prédictions
        pred_df = pd.DataFrame({
            'date_time': list(train_dates) + list(test_dates),
            'actual': np.concatenate([y_train_real, y_test_real]),
            'predicted': np.concatenate([y_train_pred_real, y_test_pred_real]),
            'dataset': ['train'] * len(y_train_real) + ['test'] * len(y_test_real)
        })

        pred_df.to_csv('predictions_cnn_lstm.csv', index=False)
        print(f"   💾 Prédictions sauvegardées")

        return results, pred_df

    except Exception as e:
        print(f"   ❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_prediction_graphs_complete(results):
    """Génère TOUS les graphiques de prédiction"""
    print("\n📈 Génération des graphiques de prédiction...")

    graphs = {}

    try:
        # 1. Courbe d'apprentissage
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(results['history']['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in results['history']:
            plt.plot(results['history']['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Courbe d\'Apprentissage - Perte', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(results['history']['mae'], label='Train MAE', linewidth=2)
        if 'val_mae' in results['history']:
            plt.plot(results['history']['val_mae'], label='Validation MAE', linewidth=2)
        plt.title('Courbe d\'Apprentissage - MAE', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('MAE', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        graphs['learning_curve'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 2. Comparaison prédictions vs réelles
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 1, 1)
        sample_size = min(100, len(results['test_dates']))
        indices = np.linspace(0, len(results['test_dates']) - 1, sample_size, dtype=int)

        dates_to_plot = indices

        plt.plot(dates_to_plot, results['y_test_actual'][indices],
                 'b-', label='Valeurs Réelles', alpha=0.7, linewidth=2)
        plt.plot(dates_to_plot, results['y_test_pred'][indices],
                 'r--', label='Prédictions', alpha=0.7, linewidth=2)

        plt.title('Comparaison des Prédictions vs Valeurs Réelles', fontsize=14, fontweight='bold')
        plt.xlabel('Échantillons', fontsize=12)
        plt.ylabel('Volume de Trafic', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.scatter(results['y_test_actual'], results['y_test_pred'],
                    alpha=0.6, s=30, c='green')

        min_val = min(results['y_test_actual'].min(), results['y_test_pred'].min())
        max_val = max(results['y_test_actual'].max(), results['y_test_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        plt.title('Prédictions vs Valeurs Réelles (Scatter Plot)', fontsize=14, fontweight='bold')
        plt.xlabel('Valeurs Réelles', fontsize=12)
        plt.ylabel('Prédictions', fontsize=12)
        plt.grid(True, alpha=0.3)

        metrics_text = f"RMSE: {results['metrics']['test_rmse']:.2f}\nMAE: {results['metrics']['test_mae']:.2f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        graphs['predictions_comparison'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 3. Distribution des erreurs
        plt.figure(figsize=(10, 6))
        errors = results['y_test_actual'] - results['y_test_pred']

        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.title('Distribution des Erreurs de Prédiction', fontsize=14, fontweight='bold')
        plt.xlabel('Erreur (Réel - Prédit)', fontsize=12)
        plt.ylabel('Fréquence', fontsize=12)
        plt.grid(True, alpha=0.3)

        mean_error = errors.mean()
        std_error = errors.std()
        error_text = f"Erreur moyenne: {mean_error:.2f}\nÉcart-type: {std_error:.2f}"
        plt.text(0.05, 0.95, error_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        graphs['error_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 4. Analyse avancée des erreurs
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.scatter(results['y_test_actual'], results['y_test_pred'],
                    alpha=0.6, s=20, c='green', edgecolors='black', linewidth=0.5)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.title('Scatter Plot', fontsize=14, fontweight='bold')
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue', density=True)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.title('Distribution des Erreurs', fontsize=14, fontweight='bold')
        plt.xlabel('Erreur')
        plt.ylabel('Densité')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(errors[:200], alpha=0.7, linewidth=1.5, color='purple')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
        plt.title('Évolution des Erreurs', fontsize=14, fontweight='bold')
        plt.xlabel('Échantillons')
        plt.ylabel('Erreur')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.boxplot(errors, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='darkblue'),
                    medianprops=dict(color='red', linewidth=2))
        plt.title('Box Plot des Erreurs', fontsize=14, fontweight='bold')
        plt.ylabel('Erreur')
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        graphs['error_analysis'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 5. Métriques de performance
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        train_errors = results['y_train_actual'] - results['y_train_pred']
        test_errors = results['y_test_actual'] - results['y_test_pred']

        plt.boxplot([train_errors, test_errors],
                    labels=['Train', 'Test'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                    medianprops=dict(color='red', linewidth=2))
        plt.title('Distribution des Erreurs: Train vs Test', fontsize=14, fontweight='bold')
        plt.ylabel('Erreur')
        plt.grid(True, alpha=0.3, axis='y')

        plt.subplot(1, 2, 2)
        metrics_names = ['RMSE Train', 'MAE Train', 'RMSE Test', 'MAE Test']
        metrics_values = [
            results['metrics']['train_rmse'],
            results['metrics']['train_mae'],
            results['metrics']['test_rmse'],
            results['metrics']['test_mae']
        ]

        colors = ['blue', 'lightblue', 'red', 'lightcoral']
        bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='black')

        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{value:.2f}', ha='center', va='bottom', fontsize=10)

        plt.title('Métriques de Performance', fontsize=14, fontweight='bold')
        plt.ylabel('Valeur')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        graphs['performance_metrics'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        print(f"   ✅ {len(graphs)} graphiques générés")

    except Exception as e:
        print(f"   ❌ Erreur lors de la génération des graphiques: {e}")
        import traceback
        traceback.print_exc()

    return graphs


# ============================================
# 5. FONCTIONS POUR GENERER LES VISUALISATIONS MATPLOTLIB
# ============================================

def generate_matplotlib_graphs_complete(df):
    """Génère TOUS les graphiques matplotlib"""
    print("🎨 Génération des visualisations matplotlib...")

    graphs = {}

    try:
        sns.set(style="whitegrid")

        # 1. Distribution du trafic
        plt.figure(figsize=(10, 6))
        sns.histplot(df['traffic_volume'], color='red', kde=True, bins=50)
        plt.title('Distribution du Volume de Trafic', fontsize=14, fontweight='bold')
        plt.xlabel('Volume de Trafic (véhicules/h)')
        plt.ylabel('Fréquence')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        graphs['dist_traffic'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 2. Distribution de la vitesse
        if 'speed' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['speed'], color='green', kde=True, bins=30)
            plt.title('Distribution de la Vitesse', fontsize=14, fontweight='bold')
            plt.xlabel('Vitesse (km/h)')
            plt.ylabel('Fréquence')
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            graphs['dist_speed'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        # 3. Heatmap de corrélation
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            correlation_matrix = numeric_df.corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Matrice de Corrélation', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            graphs['correlation_heatmap'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        # 4. Distribution météo
        if 'weather_main' in df.columns:
            plt.figure(figsize=(10, 6))
            weather_counts = df['weather_main'].value_counts()
            sns.barplot(x=weather_counts.values, y=weather_counts.index, palette='viridis')
            plt.title('Distribution des Conditions Météorologiques', fontsize=14, fontweight='bold')
            plt.xlabel('Nombre d\'occurrences')
            plt.ylabel('Condition Météorologique')
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            graphs['weather_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        # 5. Nuages vs Trafic
        if 'clouds_all' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='clouds_all', y='traffic_volume', data=df, color='lime', alpha=0.6)
            plt.title('Couverture Nuageuse vs Volume de Trafic', fontsize=14, fontweight='bold')
            plt.xlabel('Couverture Nuageuse (%)')
            plt.ylabel('Volume de Trafic')
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            graphs['clouds_vs_traffic'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        # 6. Neige vs Trafic
        if 'snow_1h' in df.columns and df['snow_1h'].sum() > 0:
            plt.figure(figsize=(10, 6))
            snow_data = df[df['snow_1h'] > 0]
            sns.scatterplot(x='snow_1h', y='traffic_volume', data=snow_data, color='orange', alpha=0.7)
            plt.title('Neige vs Volume de Trafic', fontsize=14, fontweight='bold')
            plt.xlabel('Neige (mm/h)')
            plt.ylabel('Volume de Trafic')
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            graphs['snow_vs_traffic_matplotlib'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        # 7. Pluie vs Trafic
        if 'rain_1h' in df.columns and df['rain_1h'].sum() > 0:
            plt.figure(figsize=(10, 6))
            rain_data = df[df['rain_1h'] > 0]
            sns.scatterplot(x='rain_1h', y='traffic_volume', data=rain_data, color='blue', alpha=0.7)
            plt.title('Pluie vs Volume de Trafic', fontsize=14, fontweight='bold')
            plt.xlabel('Pluie (mm/h)')
            plt.ylabel('Volume de Trafic')
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            graphs['rain_vs_traffic_matplotlib'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        # 8. Température vs Trafic
        if 'temp' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='temp', y='traffic_volume', data=df, color='red', alpha=0.6)
            plt.title('Température vs Volume de Trafic', fontsize=14, fontweight='bold')
            plt.xlabel('Température (°C)')
            plt.ylabel('Volume de Trafic')
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            graphs['temp_vs_traffic'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        print(f"✅ {len(graphs)} graphiques matplotlib générés")

    except Exception as e:
        print(f"⚠️  Erreur lors de la génération des graphiques matplotlib: {e}")

    return graphs


# ============================================
# 6. APPLICATION FLASK (COMPLÈTE)
# ============================================

@app.route("/")
def index():
    """Page principale - VERSION CORRIGÉE"""
    try:
        df = PREPARED_DATA.copy()

        print(f"\n📈 Génération du dashboard COMPLET...")
        print(f"   Colonnes: {df.columns.tolist()}")
        print(f"   Shape: {df.shape}")

        # Générer TOUS les graphiques matplotlib
        matplotlib_graphs = generate_matplotlib_graphs_complete(df)

        # Entraîner le modèle de prédiction principal
        has_predictions = False
        prediction_results = None
        predictions_df = None
        prediction_graphs = {}

        try:
            print("🔄 Entraînement du modèle CNN-LSTM principal...")
            if len(df) < 100:
                print(f"   ⚠️  Pas assez de données ({len(df)} < 100)")
                has_predictions = False
            else:
                prediction_results, predictions_df = train_prediction_model_safe(df)
                if prediction_results is not None:
                    prediction_graphs = generate_prediction_graphs_complete(prediction_results)
                    has_predictions = True
                    print("✅ Modèle principal entraîné avec succès!")
                else:
                    has_predictions = False
                    print("❌ L'entraînement du modèle principal a échoué")
        except Exception as e:
            print(f"⚠️  Erreur lors de l'entraînement du modèle principal: {e}")
            has_predictions = False

        # Entraîner le simulateur
        has_simulation = False
        simulation_predictions = None
        simulation_graphs = {}

        try:
            print("🔄 Entraînement du simulateur de trafic...")
            if len(df) >= 100:
                has_simulation = SIMULATOR.train_models()
                if has_simulation:
                    # Générer des prédictions de simulation
                    simulation_predictions = SIMULATOR.predict_next_24h()
                    if simulation_predictions is not None:
                        simulation_graphs = SIMULATOR.generate_simulation_graphs(simulation_predictions)
                        print("✅ Simulateur entraîné avec succès!")
                    else:
                        has_simulation = False
                        print("❌ Impossible de générer les prédictions de simulation")
                else:
                    print("❌ L'entraînement du simulateur a échoué")
            else:
                print(f"   ⚠️  Pas assez de données pour le simulateur ({len(df)} < 100)")
        except Exception as e:
            print(f"⚠️  Erreur lors de l'entraînement du simulateur: {e}")
            has_simulation = False

        # Statistiques principales
        stats = {
            'source': os.path.basename(DATA_SOURCE),
            'total_records': f"{len(df):,}",
            'date_range': f"{df['date_time'].min().strftime('%d/%m/%Y')} - {df['date_time'].max().strftime('%d/%m/%Y')}",
            'period_days': f"{(df['date_time'].max() - df['date_time'].min()).days} jours",
            'data_points_per_day': f"{len(df) / ((df['date_time'].max() - df['date_time'].min()).days + 1):.1f}",
            'avg_traffic': f"{df['traffic_volume'].mean():.0f}",
            'max_traffic': f"{df['traffic_volume'].max():.0f}",
            'min_traffic': f"{df['traffic_volume'].min():.0f}",
        }

        if 'speed' in df.columns:
            stats.update({
                'avg_speed': f"{df['speed'].mean():.1f}",
                'min_speed': f"{df['speed'].min():.1f}",
                'max_speed': f"{df['speed'].max():.1f}"
            })

        # Conversion température de Kelvin à Celsius
        if 'temp' in df.columns:
            # Les données semblent être en Kelvin (moyenne > 200)
            if df['temp'].mean() > 200:
                temp_celsius = df['temp'] - 273.15
                stats.update({
                    'avg_temp': f"{temp_celsius.mean():.1f}°C",
                    'min_temp': f"{temp_celsius.min():.1f}°C",
                    'max_temp': f"{temp_celsius.max():.1f}°C"
                })
            else:
                stats.update({
                    'avg_temp': f"{df['temp'].mean():.1f}°C",
                    'min_temp': f"{df['temp'].min():.1f}°C",
                    'max_temp': f"{df['temp'].max():.1f}°C"
                })

        # KPIs
        kpis = {}

        # Heure de pointe
        hourly_avg = df.groupby('hour')['traffic_volume'].mean()
        if not hourly_avg.empty:
            peak_hour = hourly_avg.idxmax()
            kpis['peak_hour'] = f"{peak_hour:02d}:00"
            kpis['peak_traffic'] = f"{hourly_avg.max():.0f}"
            kpis['lowest_hour'] = f"{hourly_avg.idxmin():02d}:00"
            kpis['hourly_variation'] = f"{(hourly_avg.max() - hourly_avg.min()):.0f}"

        # Variation hebdomadaire
        if 'day_name' in df.columns:
            daily_avg = df.groupby('day_name')['traffic_volume'].mean()
            if len(daily_avg) > 0:
                kpis['busiest_day'] = daily_avg.idxmax()
                kpis['quietest_day'] = daily_avg.idxmin()

        # Différence week-end/semaine - CORRECTION ICI
        if 'is_weekend' in df.columns:
            # CORRECTION : Utiliser .loc au lieu de l'indexation booléenne directe
            weekday_traffic = df.loc[df['is_weekend'] == 0, 'traffic_volume'].mean()
            weekend_traffic = df.loc[df['is_weekend'] == 1, 'traffic_volume'].mean()
            kpis['weekday_traffic'] = f"{weekday_traffic:.0f}"
            kpis['weekend_traffic'] = f"{weekend_traffic:.0f}"
            if weekday_traffic > 0:
                kpis['weekend_ratio'] = f"{(weekend_traffic / weekday_traffic * 100):.1f}%"

        # Graphiques Plotly
        eda_graphs = {}

        # 1. Distribution du trafic
        fig1 = px.histogram(df, x='traffic_volume', nbins=50,
                            title='Distribution du Volume de Trafic',
                            labels={'traffic_volume': 'Volume de Trafic (véhicules/h)'},
                            color_discrete_sequence=['#3498db'])
        eda_graphs['hist_traffic'] = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        # 2. Trafic par heure
        hourly_stats = df.groupby('hour').agg({
            'traffic_volume': ['mean', 'std', 'min', 'max']
        }).reset_index()
        hourly_stats.columns = ['hour', 'mean', 'std', 'min', 'max']

        fig2 = px.line(hourly_stats, x='hour', y='mean',
                       title='Trafic Moyen par Heure de la Journée',
                       labels={'hour': 'Heure', 'mean': 'Trafic Moyen'})

        fig2.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.1, line_width=0,
                       annotation_text="Heure de pointe<br>matinale", annotation_position="top left")
        fig2.add_vrect(x0=16, x1=19, fillcolor="red", opacity=0.1, line_width=0,
                       annotation_text="Heure de pointe<br>soirée", annotation_position="top right")

        fig2.update_layout(xaxis_title="Heure de la Journée", yaxis_title="Trafic Moyen")
        eda_graphs['hourly_traffic'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        # 3. Trafic par jour de semaine
        if 'day_name' in df.columns:
            weekday_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            weekly_data = df.groupby('day_name')['traffic_volume'].mean().reset_index()
            weekly_data['day_name'] = pd.Categorical(weekly_data['day_name'], categories=weekday_order, ordered=True)
            weekly_data = weekly_data.sort_values('day_name')

            fig3 = px.bar(weekly_data, x='day_name', y='traffic_volume',
                          title='Trafic Moyen par Jour de Semaine',
                          labels={'day_name': 'Jour', 'traffic_volume': 'Trafic Moyen'},
                          color='traffic_volume',
                          color_continuous_scale='Viridis')
            eda_graphs['weekly_traffic'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

        # 4. Relation vitesse-trafic
        if 'speed' in df.columns:
            sample_size = min(2000, len(df))
            sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df

            fig4 = px.scatter(sample_df, x='traffic_volume', y='speed',
                              title='Relation Vitesse vs Volume de Trafic',
                              labels={'traffic_volume': 'Volume de Trafic', 'speed': 'Vitesse (km/h)'},
                              opacity=0.6,
                              color='hour_category')
            eda_graphs['speed_vs_traffic'] = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

        # 5. Série temporelle
        if len(df) > 24:
            days_to_show = min(30, len(df) // 24)
            cutoff_date = df['date_time'].max() - pd.Timedelta(days=days_to_show)
            recent_data = df[df['date_time'] > cutoff_date]

            if len(recent_data) > 0:
                fig5 = px.line(recent_data, x='date_time', y='traffic_volume',
                               title=f'Évolution du Trafic sur {days_to_show} jours',
                               labels={'date_time': 'Date/Heure', 'traffic_volume': 'Volume de Trafic'})
                fig5.update_xaxes(rangeslider_visible=True)
                eda_graphs['timeseries'] = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

        # 6. Impact de la température
        if 'temp' in df.columns:
            # Utiliser la température en Celsius si c'est en Kelvin
            if df['temp'].mean() > 200:
                temp_for_plot = df['temp'] - 273.15
            else:
                temp_for_plot = df['temp']

            temp_bins = pd.cut(temp_for_plot, bins=10)
            temp_groups = df.groupby(temp_bins)['traffic_volume'].mean().reset_index()
            temp_groups['temp_range'] = temp_groups['temp'].astype(str)

            fig6 = px.bar(temp_groups, x='temp_range', y='traffic_volume',
                          title='Impact de la Température sur le Trafic',
                          labels={'temp_range': 'Plage de Température (°C)', 'traffic_volume': 'Trafic Moyen'})
            eda_graphs['temp_impact'] = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

        # 7. Heatmap heure/jour
        try:
            pivot_data = df.pivot_table(values='traffic_volume',
                                        index='hour',
                                        columns='day_of_week',
                                        aggfunc='mean')

            fig7 = px.imshow(pivot_data,
                             labels=dict(x="Jour (0=Lun, 6=Dim)", y="Heure", color="Trafic"),
                             title='Heatmap: Trafic par Heure et Jour',
                             color_continuous_scale='YlOrRd')
            eda_graphs['heatmap'] = json.dumps(fig7, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass

        # 8. Histogramme de la vitesse
        if 'speed' in df.columns:
            fig8 = px.histogram(df, x='speed', nbins=30,
                                title='Distribution de la Vitesse',
                                labels={'speed': 'Vitesse (km/h)'},
                                color_discrete_sequence=['#2ecc71'])
            eda_graphs['hist_speed'] = json.dumps(fig8, cls=plotly.utils.PlotlyJSONEncoder)

        # 9. Snow vs Traffic
        if 'snow_1h' in df.columns:
            snow_data = df[df['snow_1h'] > 0]
            if len(snow_data) > 0:
                fig9 = px.scatter(snow_data, x='snow_1h', y='traffic_volume',
                                  title='Impact de la Neige sur le Trafic',
                                  labels={'snow_1h': 'Neige (mm/h)', 'traffic_volume': 'Volume de Trafic'},
                                  color_discrete_sequence=['#3498db'])
                eda_graphs['snow_vs_traffic'] = json.dumps(fig9, cls=plotly.utils.PlotlyJSONEncoder)
            else:
                fig9 = px.scatter(title='Impact de la Neige sur le Trafic')
                fig9.update_layout(
                    annotations=[dict(
                        text=f"Pas de données de neige\n{len(df)} lignes, 0 avec neige",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=14)
                    )]
                )
                eda_graphs['snow_vs_traffic'] = json.dumps(fig9, cls=plotly.utils.PlotlyJSONEncoder)

        # 10. Rain vs Traffic
        if 'rain_1h' in df.columns:
            rain_data = df[df['rain_1h'] > 0]
            if len(rain_data) > 0:
                fig10 = px.scatter(rain_data, x='rain_1h', y='traffic_volume',
                                   title='Impact de la Pluie sur le Trafic',
                                   labels={'rain_1h': 'Pluie (mm/h)', 'traffic_volume': 'Volume de Trafic'},
                                   color_discrete_sequence=['#9b59b6'])
                eda_graphs['rain_vs_traffic'] = json.dumps(fig10, cls=plotly.utils.PlotlyJSONEncoder)
            else:
                fig10 = px.scatter(title='Impact de la Pluie sur le Trafic')
                fig10.update_layout(
                    annotations=[dict(
                        text=f"Pas de données de pluie\n{len(df)} lignes, 0 avec pluie",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=14)
                    )]
                )
                eda_graphs['rain_vs_traffic'] = json.dumps(fig10, cls=plotly.utils.PlotlyJSONEncoder)

        # 11. Matrice de corrélation Plotly
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            important_cols = [col for col in numeric_cols if
                              col in ['traffic_volume', 'speed', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour',
                                      'day_of_week', 'month']]

            if len(important_cols) > 1:
                corr_matrix = df[important_cols].corr()
                fig11 = px.imshow(corr_matrix,
                                  text_auto=True,
                                  title='Matrice de Corrélation',
                                  color_continuous_scale='RdBu_r',
                                  zmin=-1, zmax=1,
                                  labels=dict(color="Corrélation"))
                eda_graphs['correlation_matrix'] = json.dumps(fig11, cls=plotly.utils.PlotlyJSONEncoder)

        # Données pour le tableau
        all_columns = df.columns.tolist()
        max_columns = 8
        table_columns = all_columns[:max_columns] if len(all_columns) > max_columns else all_columns

        table_data = df[table_columns].head(15).copy()

        if 'date_time' in table_columns:
            table_data['date_time'] = table_data['date_time'].dt.strftime('%Y-%m-%d %H:%M')

        for col in table_columns:
            if col != 'date_time' and pd.api.types.is_numeric_dtype(table_data[col]):
                table_data[col] = table_data[col].round(2)
            if pd.api.types.is_string_dtype(table_data[col]):
                table_data[col] = table_data[col].astype(str).str[:30]

        hidden_columns = []
        if len(all_columns) > max_columns:
            hidden_columns = all_columns[max_columns:]

        # Données de prédiction principale
        prediction_table_data = None
        if has_predictions and predictions_df is not None:
            prediction_table_data = predictions_df.tail(15).copy()

            if 'date_time' in prediction_table_data.columns:
                try:
                    prediction_table_data['date_time'] = pd.to_datetime(prediction_table_data['date_time']).dt.strftime(
                        '%Y-%m-d %H:%M')
                except:
                    prediction_table_data['date_time'] = prediction_table_data['date_time'].astype(str)

            prediction_table_data['actual'] = prediction_table_data['actual'].round(0)
            prediction_table_data['predicted'] = prediction_table_data['predicted'].round(0)
            prediction_table_data['error'] = (
                    prediction_table_data['actual'] - prediction_table_data['predicted']).round(0)
            prediction_table_data['error_percent'] = (
                    (prediction_table_data['error'] / prediction_table_data['actual']) * 100).round(1)

        # Données de simulation
        simulation_table_data = None
        if has_simulation and simulation_predictions is not None:
            simulation_table_data = simulation_predictions.copy()
            simulation_table_data['date_time'] = simulation_table_data['date_time'].dt.strftime('%Y-%m-%d %H:%M')

        print(f"✅ Dashboard COMPLET généré!")

        return render_template("index.html",
                               stats=stats,
                               kpis=kpis,
                               eda_graphs=eda_graphs,
                               matplotlib_graphs=matplotlib_graphs,
                               prediction_graphs=prediction_graphs,
                               simulation_graphs=simulation_graphs,
                               has_predictions=has_predictions,
                               has_simulation=has_simulation,
                               prediction_metrics=prediction_results['metrics'] if has_predictions else None,
                               prediction_table_data=prediction_table_data.to_dict(
                                   'records') if prediction_table_data is not None else None,
                               simulation_table_data=simulation_table_data.to_dict(
                                   'records') if simulation_table_data is not None else None,
                               table_data=table_data.to_dict('records'),
                               table_columns=table_columns,
                               hidden_columns=hidden_columns,
                               total_columns=len(all_columns),
                               now=datetime.datetime.now().strftime("%d/%m/%Y %H:%M"))

    except Exception as e:
        import traceback
        error_msg = f"Erreur dans la génération du dashboard: {str(e)}"
        print(f"\n❌ {error_msg}")
        print(traceback.format_exc())

        return f"""
        <html>
            <head><title>Erreur</title></head>
            <body style="font-family: Arial; padding: 30px;">
                <h1 style="color: red;">⚠️ Erreur dans le Dashboard</h1>
                <div style="background: #ffe6e6; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p><strong>{error_msg}</strong></p>
                    <pre>{traceback.format_exc()}</pre>
                </div>
                <p><a href="/">Réessayer</a></p>
            </body>
        </html>
        """


@app.route("/download_predictions")
def download_predictions():
    """Télécharger les prédictions en CSV"""
    try:
        if os.path.exists('predictions_cnn_lstm.csv'):
            return send_file('predictions_cnn_lstm.csv',
                             as_attachment=True,
                             download_name='predictions_trafic.csv',
                             mimetype='text/csv')
        else:
            return "Aucun fichier de prédictions disponible."
    except Exception as e:
        return f"Erreur lors du téléchargement: {str(e)}"


@app.route("/download_simulation")
def download_simulation():
    """Télécharger les prédictions de simulation en CSV"""
    try:
        # Générer les prédictions de simulation
        if hasattr(SIMULATOR, 'models') and 'multi' in SIMULATOR.models:
            predictions = SIMULATOR.predict_next_24h()
            if predictions is not None:
                filename = 'simulation_24h.csv'
                predictions.to_csv(filename, index=False)
                return send_file(filename,
                                 as_attachment=True,
                                 download_name='simulation_trafic_24h.csv',
                                 mimetype='text/csv')

        return "Aucune simulation disponible."
    except Exception as e:
        return f"Erreur lors du téléchargement: {str(e)}"


@app.route("/api/simulation")
def api_simulation():
    """API pour obtenir les prédictions de simulation"""
    try:
        if hasattr(SIMULATOR, 'models') and 'multi' in SIMULATOR.models:
            predictions = SIMULATOR.predict_next_24h()
            if predictions is not None:
                return jsonify({
                    'success': True,
                    'predictions': predictions.to_dict('records')
                })

        return jsonify({
            'success': False,
            'error': 'Simulateur non disponible'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == "__main__":
    # Vérifier la structure
    if not os.path.exists('templates'):
        print("❌ ERREUR: Le dossier 'templates' n'existe pas!")
        print("✅ Créez un dossier 'templates' et placez index.html dedans")
    else:
        if not os.path.exists('templates/index.html'):
            print("❌ ERREUR: Le fichier 'templates/index.html' n'existe pas!")
        else:
            print("✅ Structure de dossiers OK")

    print("🌐 Accédez au dashboard: http://localhost:5000")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)