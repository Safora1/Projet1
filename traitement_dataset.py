import pyshark
import pandas as pd

# Fonction pour extraire un fichier PCAP et le convertir en DataFrame
def extract_pcap_to_dataframe(pcap_file):
    # Capture des paquets dans le fichier PCAP
    cap = pyshark.FileCapture(pcap_file)
    data = []
    
    # Itérer à travers chaque paquet dans le fichier PCAP
    for pkt in cap:
        try:
            # Vérifier que le paquet contient un en-tête IP
            if 'IP' in pkt:
                # Extraire le protocole (ICMP, TCP, UDP, etc.)
                protocol = pkt.transport_layer if hasattr(pkt, 'transport_layer') else 'Other'
                
                # Traitement des paquets ICMP seulement pour request/reply
                if 'ICMP' in pkt:
                    if hasattr(pkt.icmp, 'type'):
                        if pkt.icmp.type == '8':  # Ping request
                            request_reply = 'request'
                        elif pkt.icmp.type == '0':  # Ping reply
                            request_reply = 'reply'
                        else:
                            request_reply = 'unknown'
                    else:
                        request_reply = 'unknown'
                else:
                    request_reply = 'not_applicable'  # Pour les paquets non ICMP

                # Créer un dictionnaire pour chaque paquet
                row = {
                    'src_ip': pkt.ip.src,
                    'dst_ip': pkt.ip.dst,
                    'protocol': protocol,
                    'length': int(pkt.length),
                    'time': str(pkt.sniff_time),
                    'ttl': int(pkt.ip.ttl) if hasattr(pkt.ip, 'ttl') else None,
                    'id': int(pkt.icmp.id) if 'ICMP' in pkt else None,
                    'seq': int(pkt.icmp.seq) if 'ICMP' in pkt else None,
                    'request_reply': request_reply
                }
                data.append(row)
        except AttributeError:
            continue

    # Retourner le DataFrame avec les données extraites
    return pd.DataFrame(data)


# Extraction des deux types de trafic (normal et anormal)
df_normal = extract_pcap_to_dataframe('normal.pcap')  # Fichier PCAP normal
df_anormal = extract_pcap_to_dataframe('anormal.pcap')  # Fichier PCAP anormal

# Ajout de la colonne label pour chaque dataset
df_normal['label'] = 0  # 0 pour normal
df_anormal['label'] = 1  # 1 pour anormal

# Suppression des colonnes inutiles si nécessaire (par exemple 'time' si non pertinente)
cols_to_remove = ['time']
df_normal = df_normal.drop(columns=cols_to_remove, errors='ignore')
df_anormal = df_anormal.drop(columns=cols_to_remove, errors='ignore')

# Fusionner les deux datasets (normal et anormal) pour l'entraînement
df_final = pd.concat([df_normal, df_anormal], ignore_index=True)

# Encodage du protocole (pour le rendre numérique)
protocol_map = {'ICMP': 1, 'TCP': 2, 'UDP': 3, 'Other': 4}
df_final['protocol'] = df_final['protocol'].map(protocol_map).fillna(4)

# Vérifier les statistiques pour avoir un aperçu de l'équilibre des labels
print("Distribution des labels :")
print(df_final['label'].value_counts())

# Sauvegarder le dataset final sous forme de CSV
df_final.to_csv('dataset_final_with_request_reply.csv', index=False)

print("Données traitées et sauvegardées dans 'dataset_final_with_request_reply.csv'.")
