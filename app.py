import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import hashlib  # Pour créer l'empreinte unique de la course
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
st.set_page_config(page_title="Turf IA Pro V21", layout="wide", page_icon="🏇")
DB_FILE = "base_donnees_turf.csv"

# ==========================================
# 0. SÉCURITÉ
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state.password_correct = True
            del st.session_state["password"]
        else:
            st.session_state.password_correct = False
    if not st.session_state.password_correct:
        st.text_input("🔒 Mot de passe :", type="password", on_change=password_entered, key="password")
        return False
    return True

try:
    if st.secrets and not check_password(): st.stop()
except FileNotFoundError: pass

# ==========================================
# 1. CALCULS & MÉMOIRE
# ==========================================
def calcul_musique_severe(liste_musique):
    if not liste_musique: return 0
    score = 0
    points = {'1': 100, '2': 80, '3': 60, '4': 40, '5': 20, '0': 0, 'D': -50, 'T': -50, 'A': -50}
    for i, perf in enumerate(liste_musique[:5]):
        match = re.match(r'(\d+|[TDA])', perf)
        if match:
            el = match.group(1)
            val = points.get(el, 0)
            poids = 2.0 if i == 0 else (1.0 if i == 1 else 0.5)
            score += val * poids
    return score

def calcul_regularite(liste_musique):
    if not liste_musique: return 0
    nb = sum(1 for p in liste_musique if re.match(r'([123])', p))
    return (nb / len(liste_musique)) * 100

def get_historique_cheval(nom_cheval, df_bdd):
    if df_bdd is None or df_bdd.empty: return 0
    if not nom_cheval: return 0
    hist = df_bdd[df_bdd['Cheval'].str.upper() == nom_cheval.upper()]
    if len(hist) == 0: return 0
    return (len(hist[hist['Gagnant'] == 1]) / len(hist)) * 100

# ==========================================
# 2. SCANNER AVEC EMPREINTE DIGITALE (ID)
# ==========================================
def scanner_v21_smart(texte_brut):
    lignes = [l.strip() for l in texte_brut.strip().split('\n') if l.strip()]
    data = []
    
    df_history = pd.read_csv(DB_FILE) if os.path.exists(DB_FILE) else None
    
    # Détection Gagnant
    gagnant_detecte = 0
    match_arr = re.search(r'(?:Arrivée|Resul|Rapports|1er).*?[:\s-](?<!\d)(\d+)(?!\d)', texte_brut, re.IGNORECASE)
    if match_arr: gagnant_detecte = int(match_arr.group(1))

    current_horse = {}
    compteur_auto = 0

    for line in lignes:
        # Détection début cheval
        match_start = re.search(r'^(\d+)?\s*(.*?)\s+([HFM]\d+)$', line)
        if match_start:
            if current_horse:
                # Sauvegarde cheval précédent
                perfs = re.findall(r'\b(\d+[a-zA-Z]|[TDA][a-zA-Z]{0,2})\b', current_horse['Musique'])
                current_horse['Score_Musique'] = calcul_musique_severe(perfs)
                current_horse['Regularite'] = int(calcul_regularite(perfs))
                current_horse['Memoire_IA'] = get_historique_cheval(current_horse['Cheval'], df_history)
                current_horse['Gagnant'] = 1 if current_horse['Num'] == gagnant_detecte else 0
                data.append(current_horse)

            compteur_auto += 1
            num_txt = match_start.group(1)
            num = int(num_txt) if num_txt else compteur_auto
            raw_name = match_start.group(2)
            deferre = 0
            if 'D4' in raw_name or 'D4' in line: deferre = 2
            elif 'DA' in raw_name or 'DP' in raw_name or 'P4' in raw_name: deferre = 1
            clean_name = re.sub(r'(D4|DA|DP|P4|TrafoPA|\(BE\)|\.|Porte des oeillères)', '', raw_name).strip()
            
            current_horse = {
                'Num': num, 'Cheval': clean_name, 'Jockey': 'Inconnu', 'D4': deferre, 'Musique': '', 'Cote': 50.0
            }
            continue

        if current_horse:
            if '[' in line and 'm]' in line:
                parts = line.split('[')
                if len(parts) > 0: current_horse['Jockey'] = parts[0].strip()
                continue
            if re.search(r'\d+[am]', line) or 'Da' in line or 'Dm' in line:
                current_horse['Musique'] = line
                continue
            match_cote = re.match(r'^(\d+[.,]?\d*)$', line.replace(' ', ''))
            if match_cote:
                val = float(match_cote.group(1).replace(',', '.'))
                if val < current_horse['Cote']: current_horse['Cote'] = val

    if current_horse:
        perfs = re.findall(r'\b(\d+[a-zA-Z]|[TDA][a-zA-Z]{0,2})\b', current_horse['Musique'])
        current_horse['Score_Musique'] = calcul_musique_severe(perfs)
        current_horse['Regularite'] = int(calcul_regularite(perfs))
        current_horse['Memoire_IA'] = get_historique_cheval(current_horse['Cheval'], df_history)
        current_horse['Gagnant'] = 1 if current_horse['Num'] == gagnant_detecte else 0
        data.append(current_horse)

    df_result = pd.DataFrame(data)

    # --- CRÉATION DE L'ID UNIQUE (EMPREINTE) ---
    if not df_result.empty:
        # On colle tous les noms des chevaux pour faire une signature unique
        signature = "".join(sorted(df_result['Cheval'].astype(str).tolist()))
        # On crypte cette signature pour avoir un code unique
        id_unique = hashlib.md5(signature.encode()).hexdigest()
        df_result['ID_Course'] = id_unique

    return df_result, gagnant_detecte

# ==========================================
# 3. INTERFACE
# ==========================================
st.title("📱 Turf IA V21 (Anti-Doublon)")

tab1, tab2 = st.tabs(["📝 Scanner", "📊 Gestion Base"])

with tab1:
    st.info("Colle les partants (Zone-Turf / Geny / PMU)")
    texte_input = st.text_area("Données de la course :", height=200)
    
    if st.button("🚀 ANALYSER"):
        if texte_input:
            df_res, gagnant_auto = scanner_v21_smart(texte_input)
            if not df_res.empty:
                st.session_state['df_course'] = df_res
                st.session_state['gagnant_suggere'] = gagnant_auto
                if gagnant_auto > 0:
                    st.success(f"🎯 Arrivée détectée : N°{gagnant_auto}")
            else:
                st.error("Rien trouvé.")

    # Affichage
    if 'df_course' in st.session_state:
        df = st.session_state['df_course']
        
        # IA vs Maths
        if os.path.exists(DB_FILE) and len(pd.read_csv(DB_FILE)) > 50:
            df_hist = pd.read_csv(DB_FILE)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            cols_train = ['Score_Musique', 'Regularite', 'D4', 'Cote', 'Memoire_IA']
            df_hist = df_hist.fillna(0)
            if set(cols_train).issubset(df_hist.columns):
                model.fit(df_hist[cols_train], df_hist['Gagnant'])
                X_today = df[cols_train].fillna(0)
                probs = model.predict_proba(X_today)[:, 1]
                df['Chance %'] = (probs / probs.sum()) * 100
                msg = "🧠 Mode IA"
            else:
                msg = "⚠️ Colonnes manquantes dans l'historique"
        else:
            df['Points'] = (1/df['Cote']*200) + (df['Score_Musique']*0.4) + df['Memoire_IA']
            df.loc[df['D4']==2, 'Points'] += 15
            df['Chance %'] = (df['Points'] / df['Points'].sum()) * 100
            msg = "⚠️ Mode Maths"

        st.caption(msg)
        df_show = df[['Num', 'Cheval', 'Musique', 'Cote', 'Chance %']].sort_values(by='Chance %', ascending=False)
        st.dataframe(df_show, column_config={"Chance %": st.column_config.ProgressColumn(format="%.0f%%")}, hide_index=True, use_container_width=True)
        st.divider()

        # --- SAUVEGARDE INTELLIGENTE (UPDATE) ---
        st.subheader("⚡ Validation (Mise à jour Auto)")

        def action_sauvegarde_smart():
            num = st.session_state.choix_user
            if num != 0:
                df_new = st.session_state['df_course'].copy()
                
                # Mise à jour Gagnant
                df_new['Gagnant'] = 0
                df_new.loc[df_new['Num'] == num, 'Gagnant'] = 1
                
                # Colonnes à sauver
                cols = ['Num', 'Cheval', 'Jockey', 'D4', 'Score_Musique', 'Regularite', 'Cote', 'Memoire_IA', 'Gagnant', 'ID_Course']
                
                # LOGIQUE DE REMPLACEMENT
                if os.path.exists(DB_FILE):
                    df_old = pd.read_csv(DB_FILE)
                    
                    # Si la base contient des IDs, on cherche si la course existe déjà
                    if 'ID_Course' in df_old.columns and 'ID_Course' in df_new.columns:
                        id_courant = df_new['ID_Course'].iloc[0]
                        # 1. On SUPPRIME l'ancienne version de cette course
                        df_old = df_old[df_old['ID_Course'] != id_courant]
                    
                    # 2. On AJOUTE la nouvelle version
                    df_final = pd.concat([df_old, df_new[cols]])
                    df_final.to_csv(DB_FILE, index=False)
                else:
                    df_new[cols].to_csv(DB_FILE, index=False)
                
                st.toast(f"💾 Sauvegardé : Gagnant N°{num}", icon="✅")

        # Pré-sélection
        gagnant_suggere = st.session_state.get('gagnant_suggere', 0)
        try:
            idx = df['Num'].tolist().index(gagnant_suggere) + 1 if gagnant_suggere > 0 else 0
        except: idx = 0

        # Liste déroulante
        st.selectbox(
            "Gagnant :", 
            [0] + df['Num'].tolist(), 
            index=idx, 
            key="choix_user", 
            on_change=action_sauvegarde_smart
        )

with tab2:
    if os.path.exists(DB_FILE):
        df_bdd = pd.read_csv(DB_FILE)
        st.metric("Lignes totales", len(df_bdd))
        
        col1, col2 = st.columns([3, 1])
        with col1: st.info("Si tu as des doublons dans tes anciennes courses :")
        with col2:
            if st.button("🧹 NETTOYER"):
                # Nettoyage de secours sur les anciens doublons (sans ID)
                df_clean = df_bdd.drop_duplicates(subset=['Cheval', 'Num', 'Jockey', 'Musique'], keep='last')
                df_clean.to_csv(DB_FILE, index=False)
                st.success("Nettoyage terminé !")
                st.rerun()

        st.dataframe(df_bdd.tail(5))
        st.download_button("📥 Télécharger CSV", df_bdd.to_csv(index=False).encode('utf-8'), "turf.csv", "text/csv")
