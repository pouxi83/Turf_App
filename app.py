import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
st.set_page_config(page_title="Turf IA Universal", layout="wide", page_icon="🏇")
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

# Active la sécurité si des secrets sont configurés sur le Cloud
try:
    if st.secrets and not check_password():
        st.stop()
except FileNotFoundError:
    pass # Mode local sans mot de passe

# ==========================================
# 1. FONCTIONS DE CALCUL
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
    if not nom_cheval: return 0
    # Recherche insensible à la casse
    hist = df_bdd[df_bdd['Cheval'].str.upper() == nom_cheval.upper()]
    if len(hist) == 0: return 0
    # Calcul : % de victoire historique
    return (len(hist[hist['Gagnant'] == 1]) / len(hist)) * 100

# ==========================================
# 2. SCANNER INTELLIGENT (V19 - MULTI FORMATS)
# ==========================================
def scanner_v19_universal(texte_brut):
    lignes = [l.strip() for l in texte_brut.strip().split('\n') if l.strip()]
    data = []
    
    # Chargement Mémoire
    df_history = pd.read_csv(DB_FILE) if os.path.exists(DB_FILE) else None
    
    # Détection Gagnant (Arrivée en haut de page)
    gagnant_detecte = 0
    match_arr = re.search(r'(?:Arrivée|Resul|Rapports|1er).*?[:\s-](?<!\d)(\d+)(?!\d)', texte_brut, re.IGNORECASE)
    if match_arr: gagnant_detecte = int(match_arr.group(1))

    # Variables pour le parsing par bloc
    current_horse = {}
    compteur_auto = 0

    for line in lignes:
        # A. DÉTECTION DÉBUT CHEVAL (Format: "Nom H6" ou "1 Nom H6")
        # On cherche une ligne finissant par Sexe+Age (ex: H6, F10, M4)
        match_start = re.search(r'^(\d+)?\s*(.*?)\s+([HFM]\d+)$', line)

        if match_start:
            # Si un cheval était déjà en cours, on le sauvegarde
            if current_horse:
                # Finalisation du cheval précédent
                perfs = re.findall(r'\b(\d+[a-zA-Z]|[TDA][a-zA-Z]{0,2})\b', current_horse['Musique'])
                current_horse['Score_Musique'] = calcul_musique_severe(perfs)
                current_horse['Regularite'] = int(calcul_regularite(perfs))
                current_horse['Memoire_IA'] = get_historique_cheval(current_horse['Cheval'], df_history)
                current_horse['Gagnant'] = 1 if current_horse['Num'] == gagnant_detecte else 0
                data.append(current_horse)

            # Nouveau cheval
            compteur_auto += 1
            num_txt = match_start.group(1)
            # Si pas de numéro (format Geny), on utilise le compteur automatique
            num = int(num_txt) if num_txt else compteur_auto
            
            raw_name = match_start.group(2)
            
            # Détection Fers dans le nom
            deferre = 0
            if 'D4' in raw_name or 'D4' in line: deferre = 2
            elif 'DA' in raw_name or 'DP' in raw_name or 'P4' in raw_name: deferre = 1
            
            # Nettoyage du nom
            clean_name = re.sub(r'(D4|DA|DP|P4|TrafoPA|\(BE\)|\.|Porte des oeillères)', '', raw_name).strip()
            
            current_horse = {
                'Num': num,
                'Cheval': clean_name,
                'Jockey': 'Inconnu',
                'D4': deferre,
                'Musique': '',
                'Cote': 50.0
            }
            continue

        # B. LECTURE DES DÉTAILS (Si on est dans un bloc cheval)
        if current_horse:
            # 1. Jockey & Distance (Ligne avec crochets [2850m])
            if '[' in line and 'm]' in line:
                parts = line.split('[')
                if len(parts) > 0:
                    current_horse['Jockey'] = parts[0].strip()
                continue
            
            # 2. Musique (Ligne avec des parenthèses d'année ou des perfs)
            # Ex: 4a (25) 2a...
            if re.search(r'\d+[am]', line) or 'Da' in line or 'Dm' in line:
                current_horse['Musique'] = line
                continue
                
            # 3. Cote (Ligne avec juste des chiffres)
            match_cote = re.match(r'^(\d+[.,]?\d*)$', line.replace(' ', ''))
            if match_cote:
                val = float(match_cote.group(1).replace(',', '.'))
                # On prend la cote la plus basse si plusieurs lignes de chiffres (Cote probable vs Direct)
                if val < current_horse['Cote']:
                    current_horse['Cote'] = val

    # Sauvegarde du tout dernier cheval
    if current_horse:
        perfs = re.findall(r'\b(\d+[a-zA-Z]|[TDA][a-zA-Z]{0,2})\b', current_horse['Musique'])
        current_horse['Score_Musique'] = calcul_musique_severe(perfs)
        current_horse['Regularite'] = int(calcul_regularite(perfs))
        current_horse['Memoire_IA'] = get_historique_cheval(current_horse['Cheval'], df_history)
        current_horse['Gagnant'] = 1 if current_horse['Num'] == gagnant_detecte else 0
        data.append(current_horse)

    return pd.DataFrame(data), gagnant_detecte

# ==========================================
# 3. INTERFACE UTILISATEUR
# ==========================================
st.title("📱 Turf IA Ultimate (V20)")

tab1, tab2 = st.tabs(["📝 Scanner & Prono", "📊 Gestion Base"])

# --- ONGLET 1 : ANALYSE ---
with tab1:
    st.info("Compatible : Zone-Turf, Geny, PMU (Copier-coller complet)")
    texte_input = st.text_area("Colle les partants ici :", height=200)
    
    if st.button("🚀 ANALYSER"):
        if texte_input:
            df_res, gagnant_auto = scanner_v19_universal(texte_input)
            if not df_res.empty:
                st.session_state['df_course'] = df_res
                st.session_state['gagnant_suggere'] = gagnant_auto
                if gagnant_auto > 0:
                    st.success(f"🎯 Arrivée détectée : Le N°{gagnant_auto} a gagné !")
            else:
                st.error("Aucun cheval trouvé. Vérifie le format.")

    # Affichage Résultats
    if 'df_course' in st.session_state:
        df = st.session_state['df_course']
        
        # --- CERVEAU (IA vs MATHS) ---
        if os.path.exists(DB_FILE) and len(pd.read_csv(DB_FILE)) > 50:
            # Mode Machine Learning
            df_hist = pd.read_csv(DB_FILE)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # On vérifie que les colonnes existent
            cols_train = ['Score_Musique', 'Regularite', 'D4', 'Cote', 'Memoire_IA']
            # On nettoie les NaN au cas où
            df_hist = df_hist.fillna(0)
            
            model.fit(df_hist[cols_train], df_hist['Gagnant'])
            
            X_today = df[cols_train].fillna(0)
            probs = model.predict_proba(X_today)[:, 1]
            df['Chance %'] = (probs / probs.sum()) * 100
            msg_ia = "🧠 Mode IA (Apprentissage Actif)"
        else:
            # Mode Mathématique
            # Formule : Cote + Musique + Mémoire + Bonus D4
            df['Points'] = (1/df['Cote']*200) + (df['Score_Musique']*0.4) + df['Memoire_IA']
            df.loc[df['D4']==2, 'Points'] += 15
            df['Chance %'] = (df['Points'] / df['Points'].sum()) * 100
            msg_ia = "⚠️ Mode Mathématique (Base trop petite)"

        st.caption(msg_ia)
        
        # Tableau des Résultats
        # On affiche : Num, Cheval, Musique, Cote, Chance
        df_show = df[['Num', 'Cheval', 'Musique', 'Cote', 'Chance %']].sort_values(by='Chance %', ascending=False)
        
        st.dataframe(
            df_show, 
            column_config={
                "Chance %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                "Cote": st.column_config.NumberColumn(format="%.1f")
            },
            hide_index=True, 
            use_container_width=True
        )
        
        st.divider()
        
        # --- SECTION SAUVEGARDE ---
        gagnant_suggere = st.session_state.get('gagnant_suggere', 0)
        
        # On essaie de pré-remplir le sélecteur
        try:
            idx_defaut = df['Num'].tolist().index(gagnant_suggere) + 1 if gagnant_suggere > 0 else 0
        except:
            idx_defaut = 0

        st.subheader("💾 Enregistrer le résultat")
        choix = st.selectbox("Qui a gagné ?", [0]+df['Num'].tolist(), index=idx_defaut)
        
        if st.button("💾 SAUVEGARDER DANS LA BASE"):
            if choix != 0:
                df['Gagnant'] = 0
                df.loc[df['Num'] == choix, 'Gagnant'] = 1
                
                # Colonnes à garder
                cols = ['Num', 'Cheval', 'Jockey', 'D4', 'Score_Musique', 'Regularite', 'Cote', 'Memoire_IA', 'Gagnant']
                
                # Sauvegarde
                mode = not os.path.exists(DB_FILE)
                df[cols].to_csv(DB_FILE, mode='a', header=mode, index=False)
                st.success(f"✅ Course enregistrée ! Vainqueur : N°{choix}")
            else:
                st.warning("Merci de choisir un gagnant.")

# --- ONGLET 2 : BASE DE DONNÉES ---
with tab2:
    if os.path.exists(DB_FILE):
        df_bdd = pd.read_csv(DB_FILE)
        st.metric("Chevaux en mémoire", len(df_bdd))
        st.write("Aperçu des dernières données :")
        st.dataframe(df_bdd.tail(5))
        
        # Bouton Télécharger
        st.download_button(
            "📥 Télécharger le fichier CSV (Sauvegarde)", 
            df_bdd.to_csv(index=False).encode('utf-8'), 
            "base_donnees_turf.csv",
            "text/csv"
        )
    else:
        st.info("La base de données est vide pour l'instant.")
