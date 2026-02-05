import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
st.set_page_config(page_title="Turf IA Ultimate", layout="wide", page_icon="🏇")
DB_FILE = "base_donnees_turf.csv"

# ==========================================
# 0. SÉCURITÉ (MOT DE PASSE)
# ==========================================
def check_password():
    """Vérifie le mot de passe via les secrets Streamlit"""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state.password_correct = True
            del st.session_state["password"]
        else:
            st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.text_input("🔒 Mot de passe maître :", type="password", on_change=password_entered, key="password")
        return False
    return True

# On active la sécurité uniquement si on est sur le Cloud (si secrets détectés)
try:
    if st.secrets and not check_password():
        st.stop()
except FileNotFoundError:
    pass # En local sur ton PC, ça passe sans mot de passe

# ==========================================
# 1. FONCTIONS INTELLIGENTES
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
    """Mémoire : Cherche si le cheval a déjà gagné dans le passé"""
    if df_bdd is None or df_bdd.empty: return 0
    hist = df_bdd[df_bdd['Cheval'].str.upper() == nom_cheval.upper()]
    if len(hist) == 0: return 0
    # Calcul : % de victoire historique
    return (len(hist[hist['Gagnant'] == 1]) / len(hist)) * 100

def scanner_ultimate(texte_brut):
    lignes = [l.strip() for l in texte_brut.strip().split('\n') if l.strip()]
    data = []
    
    # 1. Chargement de la mémoire
    df_history = pd.read_csv(DB_FILE) if os.path.exists(DB_FILE) else None
    
    # 2. Auto-Détection du Gagnant dans le texte (ex: "Arrivée : 4 - ...")
    gagnant_detecte = 0
    match_arr = re.search(r'(?:Arrivée|Resul|Rapports|1er).*?[:\s-](?<!\d)(\d+)(?!\d)', texte_brut, re.IGNORECASE)
    if match_arr: gagnant_detecte = int(match_arr.group(1))

    # 3. Lecture ligne par ligne
    i = 0
    while i < len(lignes):
        ligne = lignes[i]
        match_nom = re.match(r'^(\d+)(.+)', ligne)
        
        if match_nom:
            num = int(match_nom.group(1))
            nom_brut = match_nom.group(2)
            
            # Nettoyage Nom & Fers
            deferre = 2 if 'D4' in nom_brut else (1 if 'DA' in nom_brut or 'DP' in nom_brut else 0)
            nom_cheval = re.sub(r'(D4|DA|DP|\.)', '', nom_brut).strip()
            
            infos = lignes[i+1] if i + 1 < len(lignes) else ""
            i += 1
            
            # Jockey
            noms = re.findall(r'[A-Z]\.\s?[A-Za-z]+', infos)
            jockey = noms[0] if noms else "Inconnu"
            
            # Nettoyage Collage (Chrono/Musique/Cote)
            infos = re.sub(r'(["\d])(\d+[a-zA-Z])', r'\1 \2', infos)
            infos = re.sub(r'([a-z])(\d)', r'\1 \2', infos)
            
            perfs = re.findall(r'\b(\d+[a-zA-Z]|[TDA][a-zA-Z]{0,2})\b', infos)
            musique_txt = " ".join(perfs)
            
            # Cotes
            txt_fin = re.sub(r'\(\d+\)', '', infos)
            nums = re.findall(r'(?<![a-zA-Z])(\d+[.,]?\d*|[1-9]\d+)(?![a-zA-Z])', txt_fin)
            cote = 50.0
            if nums:
                vals = []
                for c in nums[-2:]:
                    try:
                        v = float(c.replace(',', '.'))
                        if 1.0 < v < 300: vals.append(v)
                    except: pass
                if vals: cote = min(vals)
            
            # Interrogation de la Mémoire
            score_memoire = get_historique_cheval(nom_cheval, df_history)
            
            data.append({
                'Num': num, 'Cheval': nom_cheval, 'Jockey': jockey,
                'D4': deferre, 'Musique': musique_txt, 
                'Score_Musique': calcul_musique_severe(perfs),
                'Regularite': int(calcul_regularite(perfs)), 
                'Cote': cote, 'Memoire_IA': score_memoire,
                'Gagnant': 1 if num == gagnant_detecte else 0
            })
        i += 1
    return pd.DataFrame(data), gagnant_detecte

# ==========================================
# 2. INTERFACE APP
# ==========================================
st.title("📱 Turf IA Ultimate")

# Onglets pour mobile
tab1, tab2 = st.tabs(["📝 Scanner", "📊 Base de Données"])

with tab1:
    st.info("Copie toute la page Zone-Turf (Ctrl+A) et colle ici.")
    texte_input = st.text_area("Partants & Résultats :", height=150)
    
    if st.button("🚀 ANALYSER LA COURSE"):
        if texte_input:
            df_res, gagnant_auto = scanner_ultimate(texte_input)
            if not df_res.empty:
                st.session_state['df_course'] = df_res
                st.session_state['gagnant_suggere'] = gagnant_auto
                if gagnant_auto > 0: st.success(f"🎯 Gagnant détecté : N°{gagnant_auto}")
            else:
                st.error("Aucun cheval trouvé.")

    # Affichage Résultats
    if 'df_course' in st.session_state:
        df = st.session_state['df_course']
        
        # --- CERVEAU HYBRIDE ---
        # Si on a assez de données (>50), on utilise le Machine Learning
        if os.path.exists(DB_FILE) and len(pd.read_csv(DB_FILE)) > 50:
            df_hist = pd.read_csv(DB_FILE)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            # L'IA utilise la Mémoire + Stats pour apprendre
            X_train = df_hist[['Score_Musique', 'Regularite', 'D4', 'Cote', 'Memoire_IA']]
            model.fit(X_train, df_hist['Gagnant'])
            
            X_today = df[['Score_Musique', 'Regularite', 'D4', 'Cote', 'Memoire_IA']]
            probs = model.predict_proba(X_today)[:, 1]
            df['Chance %'] = (probs / probs.sum()) * 100
            msg_ia = "🧠 Mode IA (Machine Learning)"
        else:
            # Sinon formule Mathématique
            df['Points'] = (1/df['Cote']*200) + (df['Score_Musique']*0.4) + df['Memoire_IA']
            df.loc[df['D4']==2, 'Points'] += 15
            df['Chance %'] = (df['Points'] / df['Points'].sum()) * 100
            msg_ia = "⚠️ Mode Mathématique (Base trop petite)"

        st.caption(msg_ia)
        
        # Tableau Simplifié pour Mobile
        df_show = df[['Num', 'Cheval', 'Cote', 'Chance %']].sort_values(by='Chance %', ascending=False)
        st.dataframe(
            df_show, 
            column_config={"Chance %": st.column_config.ProgressColumn(format="%.0f%%")},
            hide_index=True, use_container_width=True
        )
        
        st.divider()
        
        # SAUVEGARDE
        gagnant_suggere = st.session_state.get('gagnant_suggere', 0)
        idx_defaut = df['Num'].tolist().index(gagnant_suggere) + 1 if gagnant_suggere in df['Num'].tolist() else 0
        
        choix = st.selectbox("Qui a gagné ?", [0]+df['Num'].tolist(), index=idx_defaut)
        
        if st.button("💾 ENREGISTRER"):
            if choix != 0:
                df['Gagnant'] = 0
                df.loc[df['Num'] == choix, 'Gagnant'] = 1
                # On sauvegarde tout ce qui est utile pour l'IA
                cols = ['Num', 'Cheval', 'Jockey', 'D4', 'Score_Musique', 'Regularite', 'Cote', 'Memoire_IA', 'Gagnant']
                mode = not os.path.exists(DB_FILE)
                df[cols].to_csv(DB_FILE, mode='a', header=mode, index=False)
                st.success("✅ Sauvegardé !")
            else:
                st.warning("Choisis un numéro.")

with tab2:
    if os.path.exists(DB_FILE):
        df_bdd = pd.read_csv(DB_FILE)
        st.metric("Chevaux en mémoire", len(df_bdd))
        st.dataframe(df_bdd.tail(10)[['Cheval', 'Gagnant']])
        st.download_button("Télécharger CSV", df_bdd.to_csv(index=False).encode('utf-8'), "turf.csv")
    else:
        st.info("Base vide.")
