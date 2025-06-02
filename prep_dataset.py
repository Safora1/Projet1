import pyshark
import pandas as pd

# === FONCTION GÉNÉRIQUE POUR EXTRAIRE UN FICHIER PCAP EN CSV ===
def extract_pcap_to_dataframe(pcap_file):
    cap = pyshark.FileCapture(pcap_file)
    data = []

    for pkt in cap:
        try:
            if 'IP' in pkt:
                protocol = pkt.transport_layer if hasattr(pkt, 'transport_layer') and pkt.transport_layer else 'UNKNOWN'
                row = {
                    'src_ip': pkt.ip.src,
                    'dst_ip': pkt.ip.dst,
                    'protocol': protocol,
                    'length': int(pkt.length),
                    'time': str(pkt.sniff_time)
                }
                data.append(row)
        except AttributeError:
            continue

    return pd.DataFrame(data)

# === EXTRACTION DES DEUX TRAFICS ===
df_normal = extract_pcap_to_dataframe('normal.pcap')
df_anormal = extract_pcap_to_dataframe('anormal.pcap')

# === AJOUT DU LABEL ===
df_normal['label'] = 0
df_anormal['label'] = 1

# === SUPPRESSION DES COLONNES NON UTILES (optionnel mais recommandé) ===
cols_to_remove = ['src_ip', 'dst_ip', 'time']
df_normal = df_normal.drop(columns=cols_to_remove, errors='ignore')
df_anormal = df_anormal.drop(columns=cols_to_remove, errors='ignore')

# === ENCODAGE DU PROTOCOLE ===
# Mapping : ICMP=0, TCP=1, UDP=2, UNKNOWN=-1
protocol_map = {'ICMP': 0, 'TCP': 1, 'UDP': 2, 'UNKNOWN': -1}
df_normal['protocol'] = df_normal['protocol'].map(protocol_map).fillna(-1)
df_anormal['protocol'] = df_anormal['protocol'].map(protocol_map).fillna(-1)

# === FUSION DES DEUX DATASETS ===
df_final = pd.concat([df_normal, df_anormal], ignore_index=True)

# === AFFICHER STATISTIQUES POUR CONTRÔLE ===
print("Distribution des labels :")
print(df_final['label'].value_counts())
print("\nAperçu des protocoles encodés :")
print(df_final['protocol'].value_counts())

# === SAUVEGARDE ===
df_final.to_csv('dataset_final.csv', index=False)
print("\n✅ Dataset final prêt : 'dataset_final.csv'")
