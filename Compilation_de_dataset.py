import pandas as pd
import glob

path = r"C:\Users\AWOUNANG\Downloads\archive\nouveau dossier"
 
#Récupère la liste de tous les fichiers CSV dans le dossier

all_files = glob.glob(path + "/*.csv")
dfs = []

#chargement des fichiers contenus dans le dosssier.
for f in all_files:
    try:
        df = pd.read_csv(f, sep=',', on_bad_lines='skip', low_memory=False)
        dfs.append(df)
        print(f"✅ Fichier chargé : {f} ({len(df)} lignes)")
    except Exception as e:
        print(f"❌ Erreur dans {f} : {e}")
#Fusion de tous les fichiers en un seul DataFrame
final_df = pd.concat(dfs, ignore_index=True)

final_df.to_csv("nairobi_air_data_merged.csv", index=False)

print(f"\n✅ Fusion terminée ! Nombre total de lignes : {len(final_df)}")

